function [] = q2()
kw = 0.25;  % Recommended RBF kernel width
Lambda = 10;     % Recommended box constraint
x = 10*rand(150,2);
y = (x(:,1) - 6).^2 + 3*(x(:,2) - 5).^2 - 8;
y(y > 0) = 1; y(y ~= 1) = -1;

k = 3;
Data = kfold(x,y,k);
for i = 1:k
  acc = 0;
  F = SVMtrial(Data.train.X(:,:,i),Data.train.Y(i,:)',kw,Lambda);
  sz = size(Data.test.X(:,1,i));
  sz = sz(1);
  x_for_now = Data.test.X(:,:,i);
  x_for_now_but_normy = normy(x_for_now, Data.test.Y(i,:));
  for o = 1:sz
    fx = sign(func(x_for_now_but_normy(o,:), F.xT,F.y, F.a, F.b, F.kw, F.sv));
    if (fx * Data.test.Y(i,o)) > 0
      acc = acc +1;
    end
  end
  disp(100*acc/sz)
end

%% FUNCTION TO EVALUATE ANY UNSEEN DATA, x
%  [xT,y,a,b,kw,sv] are fixed after solving the QP.
%  f(x) = SUM_{i=sv}(y(i)*a(i)*K(x,xT(i))) + b;
  function F = func(x,xT,y,a,b,kw,sv)
    K = repmat(x,size(sv)) - xT(sv,:);      % d = (x - x')
    K = exp(-sum(K.^2,2)/kw);               % RBF: exp(-d^2/kw)
    F = sum(y(sv).*a(sv).*K) + b;           % f(x)
  end

  function x = normy(x,y)
    N = length(y');                                  % Let N = no. of samples
    xm = mean(x); xs = std(x);                      % Mean and Std. Dev.
    temp = x - xm(ones(N,1),:);
    x = temp./xs(ones(N,1),:);
  end


end
