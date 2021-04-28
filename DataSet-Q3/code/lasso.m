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
w2 = zeros(m,1);

n_iter = 500;
lmbd_list = 1:20;
for j = 1:length(lmbd_list)
    lmbd = lmbd_list(j);
    for iter = 1:n_iter
            loss(iter) = sum((y-x*w2).^2+lmbd*norm(w2,1));
            grad = -2*mean(bsxfun(@times,y-x*w2,x))+lmbd*sign(w2');
            w2 = w2-lr*grad';
    end
    w(j,:)=w2;
end
for i = 1:m-1
    plot(w(:,i),'linewidth',1)
    hold on
end
xlabel('\lambda')
ylabel('covariates')
legend('takers','income','years','public','expand','rank')