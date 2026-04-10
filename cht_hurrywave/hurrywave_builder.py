"""
HurryWave model builder.

Provides :class:`HurryWaveBuilder`, a subclass of ``ModelBuilder`` that
automates the construction of a complete HurryWave model directory from a
setup-configuration dictionary: grid, bathymetry, mask, optional wave-blocking
file, and index tiles.
"""

import os

import cht_utils.fileops as fo
import geopandas as gpd
from cht_model_builder.model_builder import ModelBuilder
from pyproj import CRS

from cht_hurrywave.hurrywave import HurryWave


class HurryWaveBuilder(ModelBuilder):
    """
    Automated builder for HurryWave model directories.

    Inherits all setup-configuration parsing from ``ModelBuilder`` and adds
    HurryWave-specific build steps (grid construction, mask generation,
    optional wave-blocking, and tile generation).

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`~cht_model_builder.model_builder.ModelBuilder`.
    **kwargs
        Keyword arguments forwarded to :class:`~cht_model_builder.model_builder.ModelBuilder`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build(
        self,
        mskfile: str = "hurrywave.msk",
        depfile: str = "hurrywave.dep",
        make_mask: bool = True,
        get_bathymetry: bool = True,
        make_waveblockingfile: bool = False,
        make_tiles: bool = True,
        quiet: bool = False,
    ) -> None:
        """
        Build a complete HurryWave model from the setup configuration.

        Steps performed (each can be toggled via the boolean flags):

        1. Write ``hurrywave.inp`` with grid and input parameters.
        2. Build the quadtree mesh.
        3. Assign bathymetry (when *get_bathymetry* is ``True``).
        4. Build and write the activity mask (when *make_mask* is ``True``).
        5. Compute and write a wave-blocking file (when
           *make_waveblockingfile* is ``True``).
        6. Generate index tiles (when *make_tiles* is ``True``).

        Parameters
        ----------
        mskfile : str, optional
            Mask filename written to the model directory.  Default is
            ``"hurrywave.msk"``.
        depfile : str, optional
            Bathymetry filename written to the model directory.  Default is
            ``"hurrywave.dep"``.
        make_mask : bool, optional
            Build the activity mask.  Default is ``True``.
        get_bathymetry : bool, optional
            Assign bathymetry values to the grid.  Default is ``True``.
        make_waveblockingfile : bool, optional
            Compute and write a wave-blocking file.  Default is ``False``.
        make_tiles : bool, optional
            Generate index tiles.  Default is ``True``.
        quiet : bool, optional
            Suppress verbose output.  Default is ``False``.
        """
        crs = CRS(self.setup_config["coordinates"]["crs"])

        hw = HurryWave()

        hw.crs = crs
        hw.input.variables.x0 = self.setup_config["coordinates"]["x0"]
        hw.input.variables.y0 = self.setup_config["coordinates"]["y0"]
        hw.input.variables.dx = self.setup_config["coordinates"]["dx"]
        hw.input.variables.dy = self.setup_config["coordinates"]["dy"]
        hw.input.variables.mmax = self.setup_config["coordinates"]["mmax"]
        hw.input.variables.nmax = self.setup_config["coordinates"]["nmax"]
        hw.input.variables.rotation = self.setup_config["coordinates"]["rotation"]
        hw.input.variables.crs_name = crs.name

        if crs.is_geographic:
            hw.input.variables.crs_type = "geographic"
            hw.input.variables.crsgeo = 1
        else:
            hw.input.variables.crs_type = "projected"
            hw.input.variables.crsgeo = 0
            if "utm" in crs.name.lower():
                hw.input.variables.crs_utmzone = crs.name[-3:]
        hw.input.variables.crs_epsg = crs.to_epsg()

        for key in self.setup_config["input"]:
            setattr(hw.input.variables, key, self.setup_config["input"][key])

        # Copy hurrywave.bnd to model folder if it exists
        bnd_src = os.path.join(self.data_path, "hurrywave.bnd")
        if os.path.exists(bnd_src):
            fo.copy_file(bnd_src, self.model_path)
            hw.input.variables.bndfile = "hurrywave.bnd"

        hw.input.variables.mskfile = mskfile
        hw.input.variables.depfile = depfile

        hw.input.model.path = self.model_path
        hw.input.write()

        # Build the mesh
        hw.grid.build()

        # Assign bathymetry
        if get_bathymetry:
            hw.grid.set_bathymetry(self.bathymetry_list)
            hw.grid.write_dep_file()

        # Build the mask
        if make_mask:
            include_polygon = None
            exclude_polygon = None
            boundary_polygon = None

            if self.setup_config["mask"]["include_polygon"]:
                include_polygon = gpd.read_file(
                    self.setup_config["mask"]["include_polygon"]
                )
            if self.setup_config["mask"]["exclude_polygon"]:
                exclude_polygon = gpd.read_file(
                    self.setup_config["mask"]["exclude_polygon"]
                )
            if self.setup_config["mask"]["open_boundary_polygon"]:
                boundary_polygon = gpd.read_file(
                    self.setup_config["mask"]["open_boundary_polygon"]
                )
            hw.grid.build_mask(
                zmin=self.setup_config["mask"]["zmin"],
                zmax=self.setup_config["mask"]["zmax"],
                include_polygon=include_polygon,
                include_zmin=self.setup_config["mask"]["include_zmin"],
                include_zmax=self.setup_config["mask"]["include_zmax"],
                exclude_polygon=exclude_polygon,
                exclude_zmin=self.setup_config["mask"]["exclude_zmin"],
                exclude_zmax=self.setup_config["mask"]["exclude_zmax"],
                boundary_polygon=boundary_polygon,
                boundary_zmin=self.setup_config["mask"]["open_boundary_zmin"],
                boundary_zmax=self.setup_config["mask"]["open_boundary_zmax"],
            )
            hw.grid.write_msk_file()

        if make_waveblockingfile:
            wbl = WaveBlockingFile(model=hw)
            wblfile = "wbl_file.nc"

            wbl.build(
                hw.grid,
                bathymetry_sets=hw.bathymetry,
                roughness_sets=[None],
                mask=hw.mask,
                nr_subgrid_pixels=50,
                file_name=os.path.join(self.model_path, wblfile),
                nr_bins=36,
                quiet=False,
                showcase=False,
            )

            hw.input.sbgfile = wblfile

        # Generate index tiles
        if make_tiles:
            dem_names = []
            z_range = []
            zoom_range = []
            if (
                self.setup_config["tiling"]["zmin"] > -99990.0
                or self.setup_config["tiling"]["zmax"] < 99990.0
            ):
                z_range = [
                    self.setup_config["tiling"]["zmin"],
                    self.setup_config["tiling"]["zmax"],
                ]
                zoom_range = [
                    self.setup_config["tiling"]["zoom_range_min"],
                    self.setup_config["tiling"]["zoom_range_max"],
                ]
                for dem in self.bathymetry_list:
                    dem_names.append(dem["dataset"].name)
            hw.make_index_tiles(
                os.path.join(self.tile_path, "indices"),
                zoom_range=zoom_range,
                z_range=z_range,
                dem_names=dem_names,
            )
