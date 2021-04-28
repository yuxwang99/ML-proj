function I = f_entropy(var)

[N,K]=size(var);

for k = 1:K
    I(k) = var(:,k).'*log(var(:,k));
end

end

