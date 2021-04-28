clear;
data = load('CASE1201.mat');
[n,m] = size(data.CASE1201); 
% %Version of using intercept and coding states
for i=1:n
    data.CASE1201{i,1}= i;
end
% data.CASE1201{:,1}=str2double(data.CASE1201{:,1});
x = str2double(data.CASE1201{:,[1,3:8]});
x = [x,ones(n,1)];

% Version of using intercept but no states
% x = normalize(data.CASE1201{:,3:8});
% x = data.CASE1201{:,3:8};
% x = [x,ones(n,1)];
% m = m-1;

y = data.CASE1201{:,2};

%Implement the stepwise selection
cov = x(:,end);
Feat_choose = [m];
fun = @(w) cov'*(y-cov*w);
w = fsolve(fun,zeros(size(cov,2),1));
RSS1 = sum((y - w*cov).^2);

% %Version of not using intercept and states
% x = normalize(data.CASE1201{:,3:8});
% m = m-2;
% y = data.CASE1201{:,2};
% Feat_choose = [];
% cov = [];
% RSS1 = sum(y.^2);

% x_train = x(13:end,:);
% y_train = y(13:end);
% x_test = x(1:12,:);
% y_test = y(1:12);
% n_train = length(y_train);
% n_test = length(y_test);

%Forward selection
[Feat_choose, RSS1] = forward_selec(Feat_choose,x,y,RSS1,cov);
cov = x(:,Feat_choose);
[Feat_choose, RSS1] = forward_selec(Feat_choose,x,y,RSS1,cov);
cov = x(:,Feat_choose);
for i = 1:10
    [Feat_choose, RSS1] = backward_selec(Feat_choose,x,y,RSS1,cov);
    cov = x(:,Feat_choose);
    [Feat_choose, RSS1] = forward_selec(Feat_choose,x,y,RSS1,cov);
    cov = x(:,Feat_choose);
end

fun = @(w) cov'*(y-cov*w);
w = fsolve(fun,zeros(size(cov,2),1));

y_pred = cov*w;
%     figure
%     plot(y_pred,'linewidth',1)
%     hold on
%     grid on
%     plot(y,'linewidth',1)
%     legend('Value of predicted SAT score', 'Value of true SAT score')
%     xlabel('data index')
%     ylabel('value')
