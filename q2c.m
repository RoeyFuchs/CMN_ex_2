function [] = q2c()
rng(9)
kw = 0.25;  % Recommended RBF kernel width
Lambda = 10;     % Recommended box constraint
VALD_SIZE = 50;

vald_x = 10*rand(VALD_SIZE,2);
vald_y = (vald_x(:,1) - 6).^2 + 3*(vald_x(:,2) - 5).^2 - 8;
vald_y(vald_y > 0) = 1; vald_y(vald_y ~= 1) = -1;

vald_x = normy(vald_x, vald_y);

size_vec = [2 4 8 16 32 64 128 256 512 1024];
acc_vec = zeros(size(size_vec))';

o = 1;
for i = size_vec
  x = 10*rand(i,2);
  y = (x(:,1) - 6).^2 + 3*(x(:,2) - 5).^2 - 8;
  y(y > 0) = 1; y(y ~= 1) = -1;
  F = SVMtrial(x,y,kw,Lambda);
  acc = 0;
  for j = 1:VALD_SIZE
    fx = sign(func(vald_x(j, :), F.xT,F.y, F.a, F.b, F.kw, F.sv));
    if isnan(fx)
      fx = sign(rand(1,1)-0.5);
    end
    if (fx * vald_y(j,:)) > 0
      acc = acc +1;
    end
  end
  acc_vec(o,:) = acc/VALD_SIZE;
  o = o+1;
end

acc_vec = acc_vec*100;
plot(size_vec, acc_vec, '-o')
title('Accuracy as a function of training set size');
ylabel('% Accuracy');
xlabel('Training set size');
ylim([0 100])


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
