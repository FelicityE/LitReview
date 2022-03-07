function [nFiles] = LoadData(varargin)
  %% Options
    % data = LoadData("top_folder_name", "file_type", "file_name")
    % data = LoadData("top_folder","-type","file_type")
    % data = LoadData("top_folder","-name","file_name")
  
  % all inputs must be stings
  
  % check that topFolder and fileType is given
  switch nargin
		case 1
			topFolder = varargin{1};
      fileType = 'txt'; % if file type not given default to csv 
		case 2
			topFolder = varargin{1};
      fileType = varargin{2};
    case 3
      topFolder = varargin{1};
      if(strcmp(varargin{2},"-type"))
        fileType = varargin{3};
      elseif(strcmp(varargin{2},"-name"))
        fileName = varargin{3};
      else
        fileType = varargin{2};
        fileName = varargin{3};
      end
    otherwise
			error('Incorrect number of inputs, see the help.');
	end

  if(isdir(topFolder))
    % checking depth of directories
    curDir = topFolder;
    count = 0;
    while(isdir(curDir))
      count++;
      nextDir = ls(curDir);
      curDir = strcat(curDir,"/",nextDir(1,1:end));
    endwhile
    % length of paths
    nFolders = "";
    for(i = 1:count)
      nFolders = strcat(nFolders,"/*");
    endfor

    % getting all files
    filePath = strcat(topFolder,nFolders,".",fileType);
    allFiles = ls(filePath);
    nFiles = length(allFiles);

    % importing data notes
    % A = importdata(allFiles(i,1:end),"\t",1);
    % B = [thing{1}]

    % getting data from all files, storing in struct dataset
    dataset.nFiles = nFiles;
    for i=1:nFiles
      %new data path for each file
      tempData.(strcat("data",(num2str(i)))) = importdata(allFiles(i,1:end),"\t",1); %%%%%%%%%%%%% assumed tab for column separation
      %seting data under header struct
      for(j = 1:length(tempData.(strcat("data",(num2str(i)))).colheaders))
        dataset.(strcat("data",(num2str(i)))).([tempData.(strcat("data",(num2str(i)))).colheaders{j}]) = [tempData.(strcat("data",(num2str(i)))).data(:,j)];
      endfor
    endfor

  else 
    error('Directory not found, add top directory to working directory');
  endif

  % saving data in octive friendly dataset
  if exist('fileName', 'var')
		save(fileName, '-7', 'dataset');
	else 
    save('dataOut.mat', '-7','dataset');
  endif
end

% % testing Notes
% clear

% topFolder = "EMG_data_for_gestures-master";
% fileType = "txt";
% curDir = topFolder;
% count = 0;
% while(isdir(curDir))
%   count++;
%   nextDir = ls(curDir);
%   curDir = strcat(curDir,"/",nextDir(1,1:end));
% endwhile
% % number of file paths
% nFolders = "";
% for(i = 1:count)
%   nFolders = strcat(nFolders,"/*");
% endfor
% filePath = strcat(topFolder,nFolders,".",fileType);
% allFiles = ls(filePath);

% nFiles = length(allFiles);

% % importing data notes
% % A = importdata(allFiles(i,1:end),"\t",1);
% % B = [thing{1}]

% for i=1:nFiles
%   tempData.(strcat("data",(num2str(i)))) = importdata(allFiles(i,1:end),"\t",1);
%   for(j = 1:length(tempData.(strcat("data",(num2str(i)))).colheaders))
%     dataset.(strcat("data",(num2str(i)))).([tempData.(strcat("data",(num2str(i)))).colheaders{j}]) = [tempData.(strcat("data",(num2str(i)))).data(:,j)];
%   endfor
% endfor

