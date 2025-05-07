function create_bnd(path, time_seconds, Hm0, Tps, meanDir, dspr, wl)
% row is time, column is location

vars = {'hm0','tps','dir','dspr','wl'};

for var=vars
    var = var{1};
    if strcmp(var,'hm0')  
        data = [time_seconds Hm0];
    elseif strcmp(var,'tps')  
        data = [time_seconds Tps];
    elseif strcmp(var,'dir')  
        data = [time_seconds meanDir];
    elseif strcmp(var,'dspr')  
        data = [time_seconds dspr];
    elseif strcmp(var,'wl')  
        data = [time_seconds wl];
    end
        
        
    fid = fopen(sprintf('%s/bnd_%s.txt', path, var), 'wt');
    for i=1:size(data,1)
        for j=1:size(data,2)
        fprintf(fid, '%4.4f ', data(i,j)); 
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end





end