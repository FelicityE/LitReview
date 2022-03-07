% a = [-4,-2,-1,0,1,2,4]
% b = (a > 0) + (a == 0)
% c = a < 0;
% for(i = 1:7)
%   d(i) = sqrt(var(a.*b));
%   f(i) = sqrt(var(a.*c));
%   % if(!b(i))
%   %   % d(i) = sqrt(var(a.*b));
%   %   % f(i) = 
%   % endif
% endfor
% d
% f

% roll = 2

% % b(1:roll:length(a)) = mean(a(1:roll:end))
% [short,long] = movavg(a,2,2)

a = [1,2,3;1,2,3;1,2,3];
a = [a;a+1;a+2;a+4]
% % b = [1;1;1];
% % b = [b;b;b;b]
% % % a(b(:)==1,:)
% % % a(b(:)==1,:).^2
% % % sum(a(b(:)==1,:).^2,2)
% % % sqrt(sum(a(b(:)==1,:).^2,2))

% % c = sqrt(sum(a(b(:)==1,:).^2,2))


% % y = movfun(@(c) mean(c), c, 2)
% x = movfun(@(c) var(c), c, 2)

% for(i =1:size(a,2))
%   a(:,i:mod(i,size(a,2))+1)
% endfor

a(1:3,1)
mean(a(1:3,1))
