function checkManopt()

if ~exist('spherefactory')
    error(['cannot find Manopt.' char(10) 'download at http://manopt.org/download.html' char(10) 'add to matlab search path via addpath(genpath(''/path/to/manopt'')).']);
end

end