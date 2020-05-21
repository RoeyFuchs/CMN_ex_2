function kFolded = SVMtrial(x,y,k)
  % data = [x y];

  total_sample = size(y);
  total_sample = total_sample(1);

  num_of_sampels_in_set = floor(total_sample/k);

  permotaion = randperm(total_sample);

  xtype = size(x);
  xtype = xtype(2:end);

  for i = 1:k
    Data.X = zeros([num_of_sampels_in_set xtype k]);
    Data.Y = zeros([num_of_sampels_in_set, k]);
  end
  currentK = 1;
  cycle = 1;
  for i = permotaion
    Data.X(cycle,:,currentK) = x(i,:);
    Data.Y(cycle,currentK) = y(i,:);
    if currentK == k
      currentK = 1;
      cycle = cycle + 1;
    else
      currentK = currentK+1;
    end
  end
  kFolded.X = zeros([k*num_of_sampels_in_set-num_of_sampels_in_set xtype k]);
  kFolded.Y = zeros([num_of_sampels_in_set k]);

  index_of_k = 1:k;
  for i = 1:k
    kFolded.X(:,:,i) = Data.X(:,:,i)



  end





end
