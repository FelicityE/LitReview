clear 
load S1_E1_A1.mat

dat = [];

dat = emg;

dat = [single(stimulus), single(repetition), dat];

load S1_E2_A1.mat

dat = [dat; single(stimulus), single(repetition), emg];

load S1_E3_A1.mat

dat = [dat; single(stimulus), single(repetition), emg];


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
keep = [];

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
    keep = [keep; temp4]; 
  endfor
  
endfor

save DB2_S1_EMG.mat keep;
csvwrite("DB2_S1_EMG.csv", keep)
