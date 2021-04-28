function CI = CItest(D,X,Y,Z)
% Assume that variables are {1,2,...,p} and we have n samples
% Input:
% D: Matrix of data (with size n*p)
% X: index of first variable
% Y: index of second variable
% Z: A vector of indices for variables of the conditioning set
% output:
% CI = 1 or 0

    alpha = 0.06;
    n = size(D,1);
    c = norminv(1-alpha/2);
    DD = D(:,[X;Y;Z]);
    R = corrcoef(DD);
    P = inv(R);
    ro = -P(1,2)/sqrt(P(1,1)*P(2,2));
    zro = 0.5*log((1+ro)/(1-ro));
    if abs(zro)<c/sqrt(n-size(Z,2)-3)
        CI = 1;
    else
        CI = 0;
    end
end

