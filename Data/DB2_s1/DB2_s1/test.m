a = [ 1,1,3,6;
      1,1,4,3;
      1,2,4,7;
      0,0,7,5;
      2,1,4,7;
      2,1,5,8;
      2,2,6,3;
      0,0,3,8;];
##      3,1,6;
##      3,1,6;
##      3,2,4;
##      0,0,7;
##      4,1,2;
##      4,1,6;
##      4,2,4;]
##
b = [];

for i = 0:max(a(:,1))+1
  b = [b; a(a(:,1)==i,:)];
endfor

b

count = zeros(1,max(a(:,1))+1);

for i = 1:size(a,1)
  count(a(i,1)+1) += 1; 
endfor

smpLen = max(count)*(size(a,2)-2)
keep = [];

for i = 1:max(b(:,1))+1
  i
  temp1 = b(b(:,1)+1 == i,:)
  for j = 1:max(b(i+1,2))+1
    j
    temp1 = temp1(temp1(:,2)+1==j,:)'
    if size(temp1,2) > 0
      temp2 = [];
      for k = 1:size(b,2)
        k
        if k == 2
          k += 1;
        else
          temp2 = [temp2, temp1(k,:)]
        endif
      endfor
      if size(temp2,2) != smpLen
        dif = smpLen - size(temp2,2);
        temp2 = [zeros(1,floor(dif/2)), temp2, zeros(1,ceil(dif/2))];
      endif
      keep = [keep; temp2]
      temp1 = b(b(:,1)+1 == i,:);
    else
      temp1 = b(b(:,1)+1 == i,:);
    endif
  endfor
endfor

keep

##x = [0,4,6,6,8;5,6,5,4,8]
##y = [4,6]
##
##sampLen = 5;
##
##dif = sampLen - size(y,2)
##
##y = [zeros(1,floor(dif/2)), y, zeros(1,ceil(dif/2))]
##
##z = [x;y]

























##bears