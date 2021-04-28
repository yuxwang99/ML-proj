function [Feat_choose, RSS1] = forward_selec(Feat_choose,x,y,RSS1,cov)

Feat = 1: size(x,2);
Feat_unchoose = setdiff(Feat, Feat_choose);
for i = 1:length(Feat_unchoose)
    new_cov = [cov,x(:,Feat_unchoose(i))];
    fun = @(w) new_cov'*(y-new_cov*w);
    w = fsolve(fun,zeros(size(new_cov,2),1));
    RSS2(i) = sum((y - new_cov*w).^2);
    F(i)= (size(new_cov,1)-size(new_cov,2))*(RSS1-RSS2(i))/RSS2(i);
end
[a,idx] = max(F);
if a > finv(0.95,1,size(new_cov,1)-size(new_cov,2))
    Feat_choose = [Feat_choose,Feat_unchoose(idx)];
    RSS1 = RSS2(idx);
end
    
end

