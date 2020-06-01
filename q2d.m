function [] = q2d()
rng(9)

kw = 0.25;  % Recommended RBF kernel width

lambda_value = [0.005 0.5 5 50 500];
VALD_SIZE = 50;
samples_for_train = 10;
k = 3;
x = 10*rand(max(VALD_SIZE, samples_for_train)*k,2);
y = (x(:,1) - 6).^2 + 3*(x(:,2) - 5).^2 - 8;
y(y > 0) = 1; y(y ~= 1) = -1;
acc_per_samples = zeros(size(lambda_value))';


o = 1;
index = 1;
Data = kfold(x,y,k);
for lam = lambda_value
  Lambda = lam;
  acc_k = zeros(k,1);
  for i = 1:k
    F = SVMtrial(Data.train.X(1:samples_for_train,:,i),Data.train.Y(i,1:samples_for_train)',kw,Lambda);
    acc = 0;
    sz = size(Data.test.X(1:VALD_SIZE,1,i));
    sz = sz(1);
    x_for_now = Data.test.X(1:VALD_SIZE,:,i);
    x_for_now_but_normy = normy(x_for_now, Data.test.Y(i,1:VALD_SIZE));
    for o = 1:sz
      fx = sign(func(x_for_now_but_normy(o,:), F.xT,F.y, F.a, F.b, F.kw, F.sv));
      if (fx * Data.test.Y(i,o)) > 0
        acc = acc +1;
      end
    end
    acc_k(i,1) = (acc/sz);
    o = o+1;
  end
  acc_per_samples(index) = mean(acc_k);
  index = index+1;
end

acc_per_samples = acc_per_samples*100;
plot(lambda_value, acc_per_samples, '-o')
title('Accuracy as a function of \lambda');
ylabel('% Accuracy');
xlabel('\lambda');
acc_per_samples






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
