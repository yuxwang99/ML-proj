clear;
data = load('CASE1201.mat');
[n,m] = size(data.CASE1201); 

for i=1:n
    data.CASE1201{i,1}= i;
end
data.CASE1201{:,1}=str2double(data.CASE1201{:,1});
m = m-1;
x = data.CASE1201{:,[3:8]};
% x =str2double(data.CASE1201{:,[1,3:8]});
x = normalize(x);
x = [x,ones(n,1)];
y = data.CASE1201{:,2};
lr = 1e-2;


w = zeros(m,1);
n_iter = 500;
for iter = 1:n_iter
    loss(iter) = sum(y-x*w).^2;
    grad = -2*mean(bsxfun(@times,y-x*w,x));
    w = w-lr*grad';
end
y_pred = x*w;
figure
plot(loss,'linewidth',1);
legend('least square loss')
xlabel('iteration times')
ylabel('value')
grid on
figure
plot(y_pred,'linewidth',1)
hold on
grid on
plot(y,'linewidth',1)
legend('Value of predicted SAT score', 'Value of true SAT score')
xlabel('data index')
ylabel('value')

for i = 1:4
    %Seperate the train set and test set
    switch i
        case 1  
            x_train = x(13:end,:);
            y_train = y(13:end);
            x_test = x(1:12,:);
            y_test = y(1:12);
            n_train = length(y_train);
            n_test = length(y_test);
            data1.CASE1201 = data.CASE1201(13:end,:);
            data2.CASE1201 = data.CASE1201(1:12,:);
        case 2
            x_train = x([1:12,25:end],:);
            y_train = y([1:12,25:end]);
            x_test = x(13:24,:);
            y_test = y(13:24);
            n_train = length(y_train);
            n_test = length(y_test);
            data1.CASE1201 = data.CASE1201([1:12,25:end],:);
            data2.CASE1201 = data.CASE1201(13:24,:);
        case 3
            x_train = x([1:24,38:end],:);
            y_train = y([1:24,38:end]);
            x_test = x(25:37,:);
            y_test = y(25:37);
            n_train = length(y_train);
            n_test = length(y_test);
            data1.CASE1201 = data.CASE1201([1:24,38:end],:);
            data2.CASE1201 = data.CASE1201(25:37,:);
        case 4
            x_train = x(1:37,:);
            y_train = y(1:37);
            x_test = x(38:end,:);
            y_test = y(38:end);
            n_train = length(y_train);
            n_test = length(y_test);
            data1.CASE1201 = data.CASE1201(1:37,:);
            data2.CASE1201 = data.CASE1201(38:end,:);
    end
    %Least squares minimization
    w = zeros(m,1);
    n_iter = 1000;
    for iter = 1:n_iter
        loss(iter) = sum(y_train-x_train*w).^2;
        grad = -2*mean(bsxfun(@times,y_train-x_train*w,x_train));
        w = w-lr*grad';
    end
 
    %test the result
    y_pred = x_test*w;
    error_lse(i) = sum(abs(y_pred-y_test)./y_test)/length(y_test);
    %Lasso
    w2 = zeros(m,1);
    n_iter = 1000;
    lmbd = 5;
    for iter = 1:n_iter
        loss(iter) = sum((y_train-x_train*w2).^2+lmbd*norm(w2,1));
        grad = -2*mean(bsxfun(@times,y_train-x_train*w2,x_train))+lmbd*sign(w2');
        w2 = w2-lr*grad';
    end
%     plot(log(loss),'linewidth', 1)
%     hold on
    %test the result
    y_pred_lasso = x_test*w2;
    error_lasso(i) = sum(abs(y_pred_lasso-y_test)./y_test)/length(y_test);
    
    y_pred_step = stepwise_sel(data1,data2);
    error_step(i) = sum(abs(y_pred_step-y_test)./y_test)/length(y_test);

end
er_lse = mean(error_lse)
er_lasso = mean(error_lasso)
er_step = mean(error_step)
