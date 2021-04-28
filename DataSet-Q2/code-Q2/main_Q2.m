clear;
data = load ('dataQ2.txt');
% data = [normalize(data(:,1:7)),data(:,8)];
data = [normalize(data(:,1:7),1),data(:,8)];
x = data(:,1:7);
y = data(:,8);

x_train = x(1:400,:);
y_train = y(1:400,:);

x_test = x(401:end,:);
y_test = y(401:end,:);

%number of neural in hidden layer1
h1 = 8;
%number of neural in hidden layer2
h2 = 16;
[m,d] = size(x_train);
%Initial the weights of the first layer
W1 = rand(h1,d);
b1 = rand(h1,1);

%Initial the weights of the second layer
W2 = rand(h2,h1);
b2 = rand(h2,1);

%Initial the weights of the third layer
W3 = rand(1,h2);
b3 = rand(1);

n_iter = 100;
y_out = zeros(m,1);
for i = 1:n_iter
    %Define learning rate
    lr = 1;
%     idx = randi(m,1);
    %propogation
    for idx = 1:m
        z1 = W1*x_train(idx,:)'+b1;
        a1 = active(z1);

        z2 = W2*a1+b2;
        a2 = active(z2);

        z3 = W3*a2+b3;
        a3 = active(z3);
        %Calculate the loss
        loss_(idx) = -(y_train(idx)*log(a3)+(1-y_train(idx))*log(1-a3));
        if a3>0.5
            y_out(idx)= 1;
        end
        accr(idx) = ~xor(y_out(idx),y_train(idx));
    %     disp(loss(i))
        %Back propogation    
        grad_b3 = -(y_train(idx)/a3+(y_train(idx)-1)/(1-a3))...
            *exp(-z3)/(1+exp(-z3)).^2;
        b3_new = b3-lr*grad_b3;

        grad_w3 = grad_b3*a2;
        W3_new = W3 - lr*grad_w3';

        grad_b2 = -(y_train(idx)/a3+(y_train(idx)-1)/(1-a3))...
            *exp(-z3)/(1+exp(-z3)).^2*W3...
            *exp(-z2)/(1+exp(-z2)).^2;
        b2_new = b2-lr*grad_b2';

        grad_w2 = grad_b2'*a1';
        W2_new = W2 - lr*grad_w2;

        grad_b1 = -((y_train(idx)/a3+(y_train(idx)-1)/(1-a3))...
            *exp(-z3)/(1+exp(-z3)).^2*W3...
            .*(exp(-z2')./(1+exp(-z2').^2))*W2)'.*exp(-z1)./(1+exp(-z1)).^2;
        b1_new = b1-lr*grad_b1;

        grad_w1 = grad_b1*x_train(idx,:);
        W1_new = W1 - lr*grad_w1;

        %Update the parameters
        W1 = W1_new;
        b1 = b1_new;
        W2 = W2_new;
        b2 = b2_new;
        W3 = W3_new;
        b3 = b3_new;
    end
    loss(i) = mean(loss_);
    accur(i) = mean(accr);
    y_out = zeros(m,1);
%     output = model(x_test,W1,W2,W3,b1,b2,b3);
%     loss_te = mean(-(y_test*log(output)+(1-y_test)*log(1-output)));
end

%train error
y_pred_tr = zeros(m,1);
for i = 1:m
    z1 = W1*x_train(i,:)'+b1;
    a1 = active(z1);
    
    z2 = W2*a1+b2;
    a2 = active(z2);
    
    z3 = W3*a2+b3;
    a3 = active(z3);
    if a3>0.5
        y_pred_tr(i)= 1;
    end
end

error_tr = length(find(y_pred_tr~=y_train))/m

%test error
n_test = 200;
y_pred = zeros(n_test,1);
for i = 1:n_test
    z1 = W1*x_test(i,:)'+b1;
    a1 = active(z1);
    
    z2 = W2*a1+b2;
    a2 = active(z2);
    
    z3 = W3*a2+b3;
    a3 = active(z3);
    if a3>0.5
        y_pred(i)= 1;
    end
end
error_te = length(find(y_pred~=y_test))/n_test
plot(loss,'linewidth',1.5)
xlabel('epoch')
% ylabel('loss')
% legend('The loss of every epoch')
hold on 
plot(accur,'linewidth',1.5)
xlabel('epoch')
% ylabel('accu')
legend('The loss of every epoch','Accurancy of every epoch')
grid on 