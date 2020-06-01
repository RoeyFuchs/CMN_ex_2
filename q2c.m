function [] = q2c()
  kw = 0.25;  % Recommended RBF kernel width
  Lambda = 10;     % Recommended box constraint

  k = 3;
  vec_size = [2 4 8 16 32 64 128 256 512 1024];
  x = 10*rand(max(vec_size)*k,2);
  y = (x(:,1) - 6).^2 + 3*(x(:,2) - 5).^2 - 8;
  y(y > 0) = 1; y(y ~= 1) = -1;
  acc_per_samples = zeros(size(vec_size))';
  samples_for_valid = 50;

  index = 1;
  for v = vec_size
    Data = kfold(x,y,k);
    acc_k = zeros(k,1);
    for i = 1:k
      acc = 0;
      F = SVMtrial(Data.train.X(1:v,:,i),Data.train.Y(i,1:v)',kw,Lambda);
      sz = size(Data.test.X(1:samples_for_valid,1,i));
      sz = sz(1);
      x_for_now = Data.test.X(1:samples_for_valid,:,i);
      x_for_now_but_normy = normy(x_for_now, Data.test.Y(i,1:samples_for_valid));
      for o = 1:sz
        fx = sign(func(x_for_now_but_normy(o,:), F.xT,F.y, F.a, F.b, F.kw, F.sv));
        if (fx * Data.test.Y(i,o)) > 0
          acc = acc +1;
        end
      end
      acc_k(i,1) = (100*acc/sz);
    end
  acc_per_samples(index) = mean(acc_k);
  index = index+1;
end

plot(vec_size, acc_per_samples, '-o');
title('Accuracy as a function of training set size');
ylabel('% Accuracy');
xlabel('Training set size');
ylim([0 100])
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
