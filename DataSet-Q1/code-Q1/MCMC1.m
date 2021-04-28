%----------------------MGT-448 HW3 MCMC Algorithm------------------------
%
% MCMC algorithm on the toy dataset data2.txt
%
clc; close all; clear;
% Load data
load data2.txt
x = data2;
% Plot and check the raw data
figure;
plot(x); 
grid on; ylabel('frequency'); xlabel('data index'); title('Original Data');

%% MCMC algorithm
% initialize parameters
k = 2;                  % number of clusters
sigma = ones(1,k);      % Dirichlet parameters
a = rand(1,k);          % Gamma parameters: shape
b = rand(1,k);          % Gamma parameters: scale
alpha = rand(1,k);      % Gaussian parameters
m = rand(1,k);          % Gaussian parameters
burn_in = 500;          % number of updating iterations

% define prior: initialize rho, phi and mu
rho = 0.5*ones(1,k) + [-0.1 0.1];
phi = [1/var(x), 1/var(x)];
mu = [mean(x)-0.1, mean(x)+0.1];

% compute updated parameters
for i = 1:burn_in
    % calculate posterior of z and prior x
    temp = rho.*sqrt(phi).*exp(-0.5*phi.*(x-mu).^2);
    p_z = temp./sum(temp,2);
    n1 = find(p_z(:,1)>=p_z(:,2));
    n2 = find(p_z(:,1)<p_z(:,2));
    len_n = [length(n1), length(n2)];
    % update parameters
    sigma_star = sigma + len_n;
    rho = drchrnd(sigma_star, 1);
    
    a_star = a + len_n;
    b_star = b + [sum((x(n1,:)-mu(1)).^2), sum((x(n2,:)-mu(2)).^2)];
    phi(1) = gamrnd(a_star(1)/2,2/b_star(1));
    phi(2) = gamrnd(a_star(2)/2,2/b_star(2));
    
    alpha_star = alpha + len_n;
    m_star = (alpha.*m + [sum(x(n1,:)) sum(x(n2,:))])...
        ./(alpha + [length(n1) length(n2)]);
    mu(1) = normrnd(m_star(1),1/(alpha_star(1)*phi(1)));
    mu(2) = normrnd(m_star(2),1/(alpha_star(2)*phi(2)));
end

%% Estimation, Verification and Comparison
% visualize posterior distribution of unknown parameters
rho_samples = drchrnd(sigma_star, 1e3);
phi_samples = [gamrnd(a_star(1)/2,2/b_star(1),1000,1) ...
    gamrnd(a_star(2)/2,2/b_star(2),1000,1)];
mu_samples = [normrnd(m_star(1),1/(alpha_star(1)*phi(1)),1000,1) ...
    normrnd(m_star(2),1/(alpha_star(2)*phi(2)),1000,1)];
figure;
subplot(131); 
histogram(rho_samples); grid on; title('rho');
subplot(132); 
histogram(phi_samples(:,1)); grid on; hold on;
histogram(phi_samples(:,2)); grid on;  title('phi');
subplot(133); 
histogram(mu_samples(:,1)); grid on; hold on;
histogram(mu_samples(:,2)); grid on;   title('mu');

% compute posterior mean
disp('------------------Estimated parameter of MoG------------------------');
rho = mean(rho_samples)
phi = mean(phi_samples)
mu = mean(mu_samples)

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

% verify our finding
disp('---------------------Verifying by fitgmdist------------------------');
warning off
gm = fitgmdist(x,2)
1./gm.Sigma

%% Dirichlet distribution
% take a sample from a dirichlet distribution
function r = drchrnd(a,n)
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
end
