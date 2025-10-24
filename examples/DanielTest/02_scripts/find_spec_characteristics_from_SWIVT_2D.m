function [specs,varargout]=find_spec_characteristics_from_SWIVT_2D(sp2file)
%reads sp2 files with a single wave spectrum and outputs
%1. hm0, 2. meandir, 3. meanT 4. Tp 5. Tm_min1_0 6. directional
%spread(degrees)
%optionally also the spectrum itself as an array



spec = swan_io_spectrum(sp2file);

% dimensions
time = spec.time;
time_seconds = (time - datenum(time(1)) )*24*3600;

f = spec.frequency;
d =  spec.directions;

if isfield(spec,'lon')
    Nloc = length(spec.lon);
else
    Nloc = 1;
end

for jj=1:length(time_seconds)
for ii=1:Nloc
    % --- energy density
    vardens = sum( squeeze(spec.VaDens(ii,:,:,jj)) ,2 ) * abs(mean(diff(d)));
    
    if false
        figure; plot(f, vardens)
        
        figure;pcolor(f,d, squeeze(spec.VaDens(ii,:,:,jj))' ); shading flat; colorbar
    end
    
    % --- Hm0
    totE=trapz(f,vardens);
    hs(jj,ii)=4*sqrt(totE); %assume unit is m2/Hz
    
    % --- Tp
    ix=find(vardens==max(vardens));
    Tps(jj,ii) =1/f(ix);
    
    
    EE = squeeze(spec.VaDens(ii,:,:,jj));
    
    % --- mead wave dir
    meanDir(jj,ii) =  180/pi * atan2( trapz(f, sum(sind(d) .* EE,2) * abs(mean(diff(d))) ), trapz(f, sum(cosd(d) .* EE, 2) * abs(mean(diff(d))) ) );
    

    integral_a = trapz(f, sum(sind(d) .* EE,2) * mean(diff(d)) );
    integral_b = trapz(f, sum(cosd(d) .* EE,2) * mean(diff(d)) );
    integral_c = trapz(f, sum(EE,2) * abs(mean(diff(d))) );
    
    
    dspr(jj,ii) = sqrt( 2*(1 - sqrt( (integral_a/integral_c)^2 + (integral_b/integral_c)^2 ) ) )*180/pi;
    
 
end
end

if size(hs,1)>1
specs.hs=hs;
specs.meanDir=meanDir;
specs.Tps=Tps;
specs.dspr=dspr;
else
specs(1)=hs;
specs(2)=meanDir;
specs(3)=NaN; % todo
specs(4)=Tps;
specs(5)=NaN; % todo
specs(6)=dspr;
end



if nargout>0

VarDens.freq=f;

if isfield(spec,'x')
VarDens.LOC = [spec.x spec.y]; %!!!!! only works when there is 1 location
elseif isfield(spec,'lon')
VarDens.lon = spec.lon; 
VarDens.lat =  spec.lat; 
end

if isfield(spec,'time')
    VarDens.time =  spec.time; 
end


varargout{1}=VarDens;
end


end

