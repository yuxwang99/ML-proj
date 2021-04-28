function lambda =  lambda_ge(W,v)
N = 1000;
K = length(W);
lambda = zeros(N,K);

for k = 1:K
    for n = 1:N
        lambda(n,k)=wishrnd(W(k),v(k));
    end
end

end

