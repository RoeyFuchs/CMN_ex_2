function kFolded = kfold(x,y,k)
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

  Data.Y = Data.Y';

  kFolded.train.X = zeros([(k-1)*num_of_sampels_in_set xtype k]); % n/k * (k-1), 2, k
  kFolded.train.Y = zeros([(k-1)*num_of_sampels_in_set k])';
  kFolded.test.X = zeros([num_of_sampels_in_set xtype k]);
  kFolded.test.Y = zeros([num_of_sampels_in_set k])';


  index_of_k = 1:k;
  for i = 1:k
    kFolded.test.X(:,:,i) = Data.X(:,:,i);
    kFolded.test.Y(i,:) = Data.Y(i,:);
    m = 0;
    for j = index_of_k(index_of_k~=i)
      kFolded.train.X((m*num_of_sampels_in_set)+1:((m+1)*num_of_sampels_in_set),:,i) = Data.X(:,:,j);
      kFolded.train.Y(i,(m*num_of_sampels_in_set)+1:((m+1)*num_of_sampels_in_set)) = Data.Y(j,:);
      m = m+1;
    end
  end
end
