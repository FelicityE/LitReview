function nnFormat(varargin)  
    % This functions outputs a CSV and .mat for the data in NN format
    % Input data should have the first column as a label and the second column as the trial number
    % Because Octave is horrible, this function will not return any label that is 0. 
    % Options 
      % nnFormat(inData)
      % nnFormat(inData, "fileName")


  switch nargin
		case 1
			dat = varargin{1};
		case 2
			dat = varargin{1};
      fileName = varargin{2};
    otherwise
			error('Incorrect number of inputs, see the help.');
	end


  b = dat(dat(:,1)!=0,:);
  count = [];
  for i = 1:max(b(:,1))
    temp1 = b(b(:,1) == i,:);
    for j = 1:max(b(:,2))
      temp2 = temp1(temp1(:,2) == j,:);
      count = [count, size(temp2,1)];
    endfor
  endfor

  smpLen = max(count)*(size(b,2)-2)+1;
  ##smpLen = max(count)*(size(b,2)-2)+2; ## with trial number
  nnForm = [];

  for i = 1:max(b(:,1))
    temp1 = b(b(:,1) == i,:);
    for j = 1:max(b(:,2))
      temp2 = temp1(temp1(:,2) == j,:);
      temp3 = temp2(:,3:end)';
      temp4 = [];
      for k = 1:size(temp3,1)
        temp4 = [temp4, temp3(k,:)];
      endfor
      if size(temp4)+1 != smpLen
  ##    if size(temp4)+2 != smpLen ## with trial number
        dif = smpLen - (size(temp4,2)+1);
  ##      dif = smpLen - (size(temp4,2)+2); ##with trial number
        temp4 = [zeros(1,floor(dif/2)), temp4, zeros(1,ceil(dif/2))];
      endif
      temp4 = [i, temp4]; ## without trial number
  ##    temp4 = [i, j, temp4]; ## with trial number
      nnForm = [nnForm; temp4]; 
    endfor
    
  endfor

  if exist('fileName', 'var')
		save(fileName, '-7', 'nnForm');
    csvwrite(strcat(fileName,'.csv'), nnForm)
	else 
    save('dataOut.mat', '-7','nnForm');
    csvwrite("dataOut.csv", nnForm)
  endif
  
end