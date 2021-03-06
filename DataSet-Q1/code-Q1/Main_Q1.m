clear;
clc;
%%%%%%%%Maximize ELBO
x = load('data2.txt');
N = length(x);
K = 2;
%%%Define the initial value
beta0 = 1;
m0 = 0.5;
W0 = 10;
v0 = 10;
a0 = 2;

%Iteration to maximize the latent variable
iter = 100;
beta = [beta0,beta0];
W = [2,2];
v = [1,2];
a = [1,2];
m = [m0-1.5,m0];

for i = 1:iter
   %Update pi 
%    pi = drchrnd(a,sp);
%    E_pi = mean(log(pi));
%    E_pi = exp(E_pi);
    hat_a = sum(a);
    E_pi = psi(a)-psi(hat_a);
    E_pi = exp(E_pi);
   
   %Update lambda
%    lambda(1) = wishrnd(inv(W(1)),v(1));
%    lambda(2) = wishrnd(inv(W(2)),v(2));
%    E_Lambda = lambda_ge(W,v);
    E_Lambda = psi(v/2)+log(2)+log(abs(W));
    E_Lambda = exp(E_Lambda);
   %Update rnk
    for n = 1:N
        for k = 1:K
            term_exp(k) = -1/(2*beta(k))-0.5*v(k)*(x(n)-m(k))*W(k)*(x(n)-m(k));
            r(n,k) = E_pi(k).*sqrt(E_Lambda(k)).*exp(term_exp(k));
        end
        if sum(r(n,:))==0
            [~,id] = max(term_exp);
            r(n,:)=0.5;
%             r(n,id)=1;
        end
        r(n,:) = r(n,:)/sum(r(n,:));
    end
    Nk = sum(r);
    %Update beta
    beta = beta0+Nk;
    
    %Calculate \overline{x}
    x_line = (r'*x)'./Nk;
    %Update m
    m = (beta0*m0+ Nk.* x_line)./beta;
    
    %Calculate Sk
    Sk(1) = r(:,1)'*(x-x_line(1)).^2/Nk(1);
    Sk(2) = r(:,2)'*(x-x_line(2)).^2/Nk(2);
%     Sk = r'*(bsxfun(@minus,x,x_line')).^2./Nk;
    %Update W
    W = 1/W0 + Nk.*Sk + beta0.*Nk./(beta0+Nk).*((x_line-m0).^2);
    W = 1./W;
    %Update v
    v = v0+Nk;
    
    %Update a
    a = a0+Nk;
    
end
%Calculate the parameters and draw the distribution
rho = drchrnd(a,1000);
figure;
subplot(3,1,1);
histogram(rho(:,1)); hold on; histogram(rho(:,2))
title('Distribution of \rho');
rho = mean(rho);

phi = lambda_ge(W,v);
subplot(3,1,2);
histogram(phi(:,1)); hold on; histogram(phi(:,2))
title('Distribution of \phi');
phi = mean(phi);

mu(:,1) = normrnd(m(1),1/(beta(1)*phi(1)),[1000,1]);
mu(:,2) = normrnd(m(2),1/(beta(2)*phi(2)),[1000,1]);
subplot(3,1,3);
histogram(mu(:,1)); hold on; histogram(mu(:,2))
title('Distribution of \mu');
mu = mean(mu);

% posterior mean as estimation of unknown parameters to generate samples
covar(:,:,1) = 1/phi(1);
covar(:,:,2) = 1/phi(2);
g = gmdistribution(mu',covar,rho');
x_tilde = random(g,1000);

% visualize x tilde and original dataset x
figure;
histogram(x); grid on; ylabel('frequency'); xlabel('x');
hold on;
histogram(x_tilde);
legend('samples from real distribution','predicted distribution','Location','best');
title('Comparison on Histogram');

figure;
ecdf(x); hold on;
ecdf(x_tilde); grid on;
legend('Empirical CDF from real samples','Empirical CDF from predicted samples'...
    ,'Location','best');
title('Comparison on Empirical CDF');

%Validate our result
gm = fitgmdist(x,2)
warning off
1./gm.Sigma
