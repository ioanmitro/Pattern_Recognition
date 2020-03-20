%----PART A OF PATTERN RECOGNITION RPOJECT-----
%NAME: IOANNIS MITRO ,AEM: 2210
%NAME: GEORGIOS FRAGKIAS ,AEM:2118



x1min=2
x1max=8
n1=400					%Number of samples for class w1
x1= x1min +rand(1,n1)*(x1max-x1min)	%create a 1x400 size array in the interval [2,8] for the x-axis of the first parallelogram



y1min=1
y1max=2
y1= y1min +rand(1,n1)*(y1max-y1min)	%create a 1x400 size array in the interval [1,2] for the y-axis of the first parallelogram
p1=plot(x1,y1,'.')			%plot the samples of class w1 represented with dots inside the first parallelogram
hold on

x2min=6
x2max=8
n2=100  				%Number of samples for class w2
x2= x2min +rand(1,n2)*(x2max-x2min)	%create a 1x400 size array with random values in the interval [6,8] for the x-axis of the second parallelogram


y2min=2.5
y2max=5.5
y2= y2min +rand(1,n2)*(y2max-y2min)	%create a 1x400 size array with random values in the interval [2.5,5.5] for the y-axis of the second parallelogram
p2=plot(x2,y2,'.')			%plot the samples of class w2 represented with dots inside the first parallelogram


%create the first parallelogram as polygon matching  x  with y coordinates of each edge and piecing them together
xparal1=[2 8 8 2 2]
yparal1=[1 1 2 2 1]
p3=plot(xparal1,yparal1)		%plot the first parallelogram
 
%create the second parallelogram as polygon matching  x  with y coordinates of each edge and piecing them together
xparal2=[6 8 8 6 6]	
yparal2=[2.5 2.5 5.5 5.5 2.5]
p4=plot(xparal2,yparal2) 		%plot the second parallelogram

xlabel('X-axis of parallelogram','FontSize',10,'FontWeight','bold','Color','k')
ylabel('Y-axis of parallelogram','FontSize',10,'FontWeight','bold','Color','k')


legend([p1 p2 p3 p4],{'Samples N1','Samples N2','w1','w2'},'Location','northwest')


hold off


%-------------------PART B--------------------

%------------------B1---------------------

A=[x1;y1]	%create an array which contains samples of class w1
B=[x2;y2]	%create an array which contains samples of class w2


%alternative way of calculatiing

mean1 = mean(A);	%create the mu matrix
Sigma1 = std(A);	%create matrix S
mean2 = mean(B);
Sigma2 = std(B);

X1 = mvnrnd(mean1,Sigma1,400);	%generate X1
X2 = mvnrnd(mean2,Sigma2,100);	%generate X2


[m,n] = size(X1);
estimated_mean1 = sum(X1)/m;
tmp1=zeros(m,n);
for i=1:n
tmp1(:,i)= ((X1(:,i) - estimated_mean1(i)));
end
covar1 = (tmp1.'*tmp1)/m;

%calculate the estimated mean value and the covariance value using maximum likelihood estimation for class w2
[h,w] = size(X2);
estimated_mean2 = sum(X2)/h;
tmp2=zeros(h,w);
for i=1:w
tmp2(:,i)= ((X2(:,i) - estimated_mean2(i)));
end
covar2 = (tmp1.'*tmp1)/h;

%i should print the estimated mean_1,estimated_mean2 and covar1,covar2 in order to see tha matrixes's values
%compute the ML estimates of mean1 and Sigma1 and mean2 and Sigma2 respectively using Gaussian_ML_estimate function
[m_hat1, S_hat1]=Gaussian_ML_estimate(X1)
[m_hat2, S_hat2]=Gaussian_ML_estimate(X2)
P=[0.8 0.2]	%Propability matrix of N1,N2





%i should print the estimated mean_1,estimated_mean2 and covar1,covar2 in order to see tha matrixes's values
%Note that the returned values depend on the
%initialization of the random generator (involved in function mvnrnd), so there is a slight deviation
%among experiments


%--------------------B2-----------------------------------
%For the Euclidean distance classifier,we use the ML estimates of the means to classify the data
%vectors of X1 and X2 respectively,where z_euclidean is an N -dimensional vector containing the labels of the classes where the
%respective data vectors are assigned by the Euclidean classifier

z_euclidean1=euclidean_classifier(m_hat1,X1)
z_euclidean2=euclidean_classifier(m_hat2,X2)


%error probabilities for the Euclidean classifier
err_euclidean1 = (1-length(find(n==z_euclidean1))/length(n));
err_euclidean2 = (1-length(find(w==z_euclidean2))/length(w));

%------------------B3-----------------------------------
%Similarly for the Mahalanobis distance classifier


z_mahalanobis1=mahalanobis_classifier(m_hat1,S_hat1,X1)
z_mahalanobis2=mahalanobis_classifier(m_hat2,S_hat2,X2)


%-------------------B4----------------------------------
%For the Bayesian classifier, use function bayes_classifier and provide as input the matrices mean,
%Sigma, P, which were used for the data set generation. In other words, use the true values of mean, Sigma, and P
%and not their estimated values


z_bayesian1=bayes_classifier(m_hat1,S_hat1,P,X1)
z_bayesian2=bayes_classifier(m_hat2,S_hat2,P,X2)


%error probabilities for the  Mahalanobis and Bayesian classifiers
err_mahalanobis1 = (1-length(find(n==z_mahalanobis1))/length(n));
err_mahalanobis2 = (1-length(find(w==z_mahalanobis2))/length(w));
err_bayesian1 = (1-length(find(n==z_bayesian1))/length(n));
err_bayesian2 = (1-length(find(w==z_bayesian2))/length(w));


%i should compare the results in order to conclude about the error propabilities







%-------------------PARTC -------------------------
%------------------PARTC1----------------------
%generate data set X1,X2 and the vector y1,y2 respectively


[l1,l2]=size(S_hat1)
mv1= mean1';
N1=400;
X1=[mvnrnd(mv1(:,1),Sigma1,N1)]';
y1=[ones(1,N1)];

%compute the eigenvalues/eigenvectors and variance percentages
M=1;
[eigenval,eigenvec,explained,Y,mean_vec]=pca_fun(X1,M);


[l3,l4]=size(Sigma2)
mv2= mean2';
N2=100;
X2=[mvnrnd(mv2(:,1),Sigma2,N2)]';
y2=[ones(1,N2)];
[eigenval2,eigenvec2,explained2,Y2,mean_vec2]=pca_fun(X2,M);


%Plot of X1
%The projections of the data points of X1 along the direction of the first principal component
%are contained in the first row of Y, returned by the function pca_ fun
figure(2), hold on
figure(2), p5=plot(X1(1,y1==1),'r.')
%Computation of the projections of X1
w=eigenvec(:,1);
t1=w'*X1(:,y1==1);
X_proj1=[t1;t1]*((w/(w'*w))*ones(1,length(t1)));
%Plot of the projection of X1
figure(3), p6=plot(X_proj1(1,:),'k.')
figure(3), axis equal
%Plot of the eigenvectors for class w1
figure(4), line([0; eigenvec(1,1)], [0; eigenvec(2,1)],'Color','green')



%Plot of X2
%The projections of the data points of X2 along the direction of the first principal component
%are contained in the first row of Y, returned by the function pca_ fun
figure(2), hold on
figure(3), hold on
figure(4), hold on
    
figure(2), p7=plot(X2(1,y2==1),'g.')
%Computation of the projections of X2
w_new=eigenvec2(:,1);
t2_new=w_new'*X2(:,y2==1);
X_proj2_new=[t2_new;t2_new]*((w_new/(w_new'*w_new))*ones(1,length(t2_new)));
%Plot of the projection of X2
figure(3), p8=plot(X_proj2_new(1,:),'r.')
figure(3), axis equal
%Plot of the eigenvectors fow class w2
figure(4), line([0; eigenvec2(1,1)], [0; eigenvec2(2,1)],'Color','red')



legend([p5,p7],{'The projections of the data points of X1','The projections of the data points of X2'},'Location','northwest')
legend([p6,p8],{'The projection of X1','The projection of X2'},'Location','northwest')
legend({'eigenvectors for class w1','eigenvectors for class w1'},'Location','northwest')

%-------------PART C2----------------------
%%error probabilities for the Euclidean classifier for the produced data after PCA performed

z_euclidean1_2=euclidean_classifier(m_hat1,X1)
z_euclidean2_2=euclidean_classifier(m_hat2,X2)

err_euclidean1_2 = (1-length(find(y1==z_euclidean1_2))/length(y1));
err_euclidean2_2 = (1-length(find(y2==z_euclidean2_2))/length(y2));


%--------------PART C3----------------------------
%In this part,we are going to apply LDA 
%estimate the mean vectors of each class using the available samples,

mv_est1(:,1)=mean(X1(:,y1==1)')';


mv_est2(:,1)=mean(X2(:,y2==1)')';


%ompute the within-scatter matrix S w , use the scatter_mat function, which computes the within
%class (Sw ), the between class (Sb), and the mixture class (Sm )  for a c-class
%classification problem based on a set of data vector

[Sw1,Sb1,Sm1]=scatter_mat(X1,y1);
[Sw2,Sb2,Sm2]=scatter_mat(X2,y2);

%Since the two classes are not equiprobable, the direction w along which Fisher's discriminant ratio is
%maximized is computed as follows
w1=inv(Sw1)*(mv_est1(:,1))
w2=inv(Sw2)*(mv_est2(:,1))


%Computation of the new projections
t1=w1'*X1(:,y1==1);


t2=w2'*X2(:,y2==1);
X_proj1_1=[t1;t1]*((w1/(w1'*w1))*ones(1,length(t1)));
X_proj2_2=[t2;t2]*((w2/(w2'*w2))*ones(1,length(t2)));
figure(5), hold on
figure(5), p9=plot(X_proj1_1(1,y1==1),'y.')
legend(p9,{'The projections of X1(LDA)'},'Location','northwest')

figure(6), hold on
figure(6), p10=plot(X_proj2_2(1,y2==1),'r.')


legend(p10,{'The projections of X2(LDA)'},'Location','northwest')







%-----------PART C4---------------------------------
z_euclidean1_3=euclidean_classifier(m_hat1,X1)
z_euclidean2_3=euclidean_classifier(m_hat2,X2)

err_euclidean1_3 = (1-length(find(y1==z_euclidean1_3))/length(y1));

err_euclidean2_3 = (1-length(find(y2==z_euclidean2_3))/length(y2));

%Comparing the result depicted in MATLAB figure 1, which was produced by the execution
%of the previous code, to the corresponding result obtained by the PCA analysis, it is readily observed
%that the classes remain well separated when the vectors of X2 are projected along the w direc-
%tion that results from Fisher's discriminant analysis. In contrast, classes were heavily overlapped
%when they were projected along the principal direction provided by PCA


%-----------------PART D---------------------------------

%To augment the data vectors of X1 by an additional coordinate that equals +1, and to change
%the class labels from 1, 2 (used before) to -1, +1, respectively


z1=[ones(1,fix(N1/2)) 2*ones(1,N1-fix(N1/2))];
X1_1=[X1; ones(1,sum(N1))];
y1_1=2*z1-3;

%The set X2 is treated similarly. To compute the classification error of the LS classifier based on X2

[w]=SSErr(X1_1,y1_1,0);
SSE_out=2*(w'*X1_1>0)-1;
err_SSE=sum(SSE_out.*y1_1<0)/sum(N1)


z2=[ones(1,fix(N2/2)) 2*ones(1,N2-fix(N2/2))];
X2_1=[X2; ones(1,sum(N2))];
y2_1=2*z2-3;

%The set X2 is treated similarly. To compute the classification error of the LS classifier based on X2

[w2]=SSErr(X2_1,y2_1,0);
SSE_out_2=2*(w2'*X2_1>0)-1;
err_SSE_2=sum(SSE_out_2.*y2_1<0)/sum(N2)












