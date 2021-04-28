%Structure learning
%PC1
clear
% load 'D1.mat'
load 'D2.mat'
% load 'D3.mat'
% warning off
[map1,n1] = PC1(D);
g1 = graph(map1, 'OmitSelfLoops');
[map2,n2] = PC2(D);
g2 = graph(map2, 'OmitSelfLoops');
[map3,n3] = PC3(D);
g3 = graph(map3, 'OmitSelfLoops');

figure
plot(g1)
figure
plot(g2)
figure
plot(g3)

function [map,t_ci] = PC1(D)
[~,p] = size(D);
id = 1:p;
map = zeros(p);
t_ci = 0;
for i = 1:p-1
    for j = i+1:p
        id_z = setdiff(id,[i,j]);
        len1 = length(id_z);
%         CI0 = CItest(D,i,j,[]);
%         if CI0 == 1
%             break
%         end
        CI = [];
        for m = 0:len1
            set = combntns(id_z,m);
            [len2, ~] = size(set);
            for n = 1:len2
                flag = CItest(D,i,j,set(n,:)');
                CI = [CI,flag];
                t_ci = t_ci+1;
                if flag == 1
                    break;
                end
            end
            if flag == 1
                break;
            end
        end
        if all(CI==0)
            map(i,j)=1;
            map(j,i)=1;
        end
    end
end
end

%PC2
% Start from a complete graph
function [map2,t_test] = PC2(D)
[~,p] = size(D);
map2 = ~diag(ones(1,p));
% for i = 1:p
%     set_j = setdiff(1:p,i);
%     for m =1:length(set_j)
%         j = set_j(m);
t_test = 0;
for i = 1:p-1
    for j = i+1:p
        %Find the neighbors
        neigh_i = find(map2(:,i)~=0);
        neigh_j = find(map2(:,j)~=0);
        neigh = setdiff(union(neigh_i,neigh_j),[i,j]);
        len1 = length(neigh);
        CI = [];
        for m = 0:len1
            set = combntns(neigh,m);
            [len2, ~] = size(set);
            for n = 1:len2
                flag = CItest(D,i,j,set(n,:)');
                CI = [CI,flag];
                t_test = t_test+1;
                if flag == 1
                    map2(i,j)=0;
                    map2(j,i)=0;
                    break;
                end
            end
            if flag == 1
                break;
            end
        end
    end
end
end

%PC3 find the moral graph
function [map3,t_test] = PC3(D)
[~,p] = size(D);
id = 1:p;
map_mr = zeros(p);
t_test = 0;
for i = 1:p-1
    for j = i+1:p
        id_z = setdiff(id,[i,j]);
        CI = CItest(D,i,j,id_z');
        t_test = t_test+1;
        if CI==0
            map_mr(i,j)=1;
            map_mr(j,i)=1;
        end
    end
end
map3 = map_mr;
[x, y] = find(triu(map_mr)==1);
xy = [x,y];
for k = 1:length(xy)
%     i = x(l);
%     for k = 1:length(y)
%         j = y(k);
        i = xy(k,1);
        j = xy(k,2);
        neigh_i = find(map3(:,i)~=0);
        neigh_j = find(map3(:,j)~=0);
        neigh = setdiff(union(neigh_i,neigh_j),[i,j]);
        len1 = length(neigh);
        CI = [];
        for m = 0:len1
            set = combntns(neigh,m);
            [len2, ~] = size(set);
            for n = 1:len2
                flag = CItest(D,i,j,set(n,:)');
                CI = [CI,flag];
                t_test = t_test+1;
                if flag == 1
                    map3(i,j)=0;
                    map3(j,i)=0;
                    break;
                end
            end
            if flag == 1
                break;
            end
        end
    end

end




