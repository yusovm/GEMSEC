clear all; close all; clc;

xi=linspace(-10,10,400);
t=linspace(0,4*pi,200);
dt=t(2)-t(1);
[Xgrid,T]=meshgrid(xi,t);

f1=sech(Xgrid+3).*(1*exp(i*2.3*T));
f2=(sech(Xgrid).*tanh(Xgrid)).*(2*exp(i*2.8*T));
f=f1+f2;

%% DMD
X=f.';
X1=X(:,1:end-1);
X2=X(:,2:end);

r=2;
[U,S,V]=svd(X1,'econ');
Ur=U(:,1:r);
Sr=S(1:r,1:r);
Vr=V(:,1:r);

Atilde=Ur'*X2*Vr/Sr;
[W,D]=eig(Atilde);
Phi=X2*Vr/Sr*W;

lambda=diag(D);
omega=log(lambda)/dt;

x1=X(:,1);
b=Phi\x1;
time_dynamics=zeros(r,length(t));
for iter=1:length(t)
    time_dynamics(:,iter)=(b.*exp(omega*t(iter)));
end
X_dmd=Phi*time_dynamics;

