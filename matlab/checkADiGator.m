function checkADiGator()

if ~exist('adigatorError')
    error(['cannot find ADiGator.' char(10) 'download at http://adigator.sourceforge.net/' char(10) 'add to matlab search path via addpath(genpath(''/path/to/adigator'')).']);
end

end