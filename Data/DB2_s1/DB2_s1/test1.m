a = [ 1,1,3,6;
      1,1,4,3;
      1,2,4,7;
      0,0,7,5;
      2,1,4,7;
      2,1,5,8;
      2,2,6,3;
      2,2,6,5;
      2,2,6,8;
      0,0,3,8;];

b = a(a(:,1)!=0,:);
count = [];

for i = 1:max(b(:,1))
  temp1 = b(b(:,1) == i,:);
  for j = 1:max(b(:,2))
    temp2 = temp1(temp1(:,2) == j,:);
    count = [count, size(temp2,1)];
  endfor
endfor

smpLen = max(count)*(size(b,2)-2)+1; ##
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
    if size(temp4)+1 != smpLen ##
      dif = smpLen - (size(temp4,2)+1); ##
      temp4 = [zeros(1,floor(dif/2)), temp4, zeros(1,ceil(dif/2))];
    endif
    temp4 = [i, temp4]; ##
    keep = [keep; temp4];
  endfor
  
endfor

keep