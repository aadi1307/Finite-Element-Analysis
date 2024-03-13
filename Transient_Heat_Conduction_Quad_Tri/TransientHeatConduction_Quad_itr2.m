clear all;
clc;

%% Defining Constants and Parameters (opted from reference code)

W=1;  %%% width
H=1;   %%% height
kappa_bar=385;
rho=6000;
rho_c=3.8151e6;
kappa=kappa_bar*eye(2);    %%% Conductivity tensor

%% Parameters for Time Integration DEtAils for Incremental Time Advancement Scheme
alpha=0;  % 0 for forward, 0.5 for central, 1 for backward
tSteps=1000;  % number of time steps

%%  Determination of Integration Points for Numerical Quadrature(options are 1, 2, or 3)
wq=3;

%% Construction of the Finite Element Mesh
%%%% number of elements in x and y directions
nElx=20;
nEly=20;
nNodeEl=4;
[nEl,nNode,EleConnArr,xGlobalC]=createmesh(nElx,nEly,W,H);

%% Dirichlet BC
% Initialize Dirichlet boundary conditions array
for i=1:nEly+1
    dirich(i,1:2)=[1+(i-1)*(nElx+1), 300];
end   %left edge
for j=i+1:2*nEly+2
    dirich(j,1:2)=[(j-i)*(nElx+1), 310];
end  % right edge
nDirich=length(dirich);

%% Set the initial condition for each node based on its x-coordinate
for i=1:nNode
    if(xGlobalC(i,1)<0.5)
        u0(i)=300;
    end
    if(xGlobalC(i,1)>=0.5)
        u0(i)=300+20*(xGlobalC(i,1)-0.5);
    end
end

%% K, M and f Matrices global and Local
% Obtain Gauss quadrature points and weights for Quad elements

gaussMatrix=quadrature(wq);
kGlobal=zeros(nNode,nNode);
mGlobal=zeros(nNode, nNode);
fGlobal=zeros(nNode,1);

for e=1:nEl
    kLocal=zeros(nNodeEl,nNodeEl);
    fLocal=zeros(nNodeEl,1);
    mLocal=zeros(nNodeEl,nNodeEl);
    
    for wqX=1:wq
        for wqY=1:wq
            [N,dNdz,dNde] = ShPF(gaussMatrix(wqX,1),gaussMatrix(wqY,1));
            J=[dNdz*xGlobalC(EleConnArr(e,:),1)   dNde*xGlobalC(EleConnArr(e,:),1) ; dNdz*xGlobalC(EleConnArr(e,:),2)   dNde*xGlobalC(EleConnArr(e,:),2)];
            dNdX=[dNdz' dNde']*inv(J);
            kLocal=kLocal + dNdX*(kappa)*dNdX'*det(J)*gaussMatrix(wqX,2)*gaussMatrix(wqY,2);
            mLocal=mLocal + N'*N*rho_c*det(J)*gaussMatrix(wqX,2)*gaussMatrix(wqY,2);
            fLocal=fLocal+0;
        end
    end
%%% Assembly
    kGlobal(EleConnArr(e,:),EleConnArr(e,:))=kGlobal(EleConnArr(e,:),EleConnArr(e,:)) + kLocal;
    fGlobal(EleConnArr(e,:),1)=fGlobal(EleConnArr(e,:),1)+fLocal;
    mGlobal(EleConnArr(e,:),EleConnArr(e,:))=mGlobal(EleConnArr(e,:),EleConnArr(e,:)) + mLocal;    
end   %%% gives assembled K, M, F matrices

%% kDirichlet, mDirichlet and fDirichlet

kDirichlet=zeros(nNode-nDirich,nNode-nDirich);
fDirichlet=zeros(nNode-nDirich,1);
mDirichlet=zeros(nNode-nDirich,nNode-nDirich);
nonDirichnodes=zeros(nNode-nDirich,1);
j=1;
for i=1:nNode
    flag=0;
    for k=1:nDirich
        
        if(i==dirich(k,1))
            flag=1;
        end
        
    end
    
    if(flag==0)
        nonDirichnodes(j)=i;
        j=j+1;
    end
end   % searching non-Dirichlet nodes

for i=1:length(nonDirichnodes)
    for j=1:length(nonDirichnodes)
        kDirichlet(i,j)=kGlobal(nonDirichnodes(i),nonDirichnodes(j));
    end
end  % kDirichlet

for i=1:length(nonDirichnodes)
    for j=1:length(nonDirichnodes)
        mDirichlet(i,j)=mGlobal(nonDirichnodes(i),nonDirichnodes(j));
    end
end  % mDirichlet

% F Dirichlet
f_dash=zeros(length(nonDirichnodes),1);
for i=1:length(nonDirichnodes)
    for j=1:nDirich
        f_dash(i)=kGlobal(nonDirichnodes(i),dirich(j,1))*dirich(j,2) + f_dash(i);
    end
end  % calculating f_dash

fDirichlet=fGlobal(nonDirichnodes)-f_dash;

%%
% Calculate the time step size

tDelta=tDelta_calculate(alpha, mDirichlet, kDirichlet);

% Initialize global displacement with initial condition
dGlobal(:,1)=u0;
d(:,1)=dGlobal(nonDirichnodes);
v(:,1)=mDirichlet\(fDirichlet - kDirichlet*d(:,1));

for i=1:tSteps
    d_tilde=d(:,i) + (1-alpha)*tDelta*v(:,i);    
    v(:,i+1)= (mDirichlet + alpha*tDelta*kDirichlet)\(fDirichlet - kDirichlet*d_tilde);    
    d(:,i+1) = d_tilde + alpha*tDelta*v(:,i+1);     
end

dGlobal(dirich(:,1))=dirich(:,2);
dGlobal(nonDirichnodes)=d(:,i);  % result at the end of tSteps

%% fem to vtu
fem_to_vtk ('Quad', xGlobalC, EleConnArr, dGlobal);

%% functions

function [N, dNdz,dNde]=ShPF(Zi,EtA)

N(1)=(1-Zi)*(1-EtA)/4;
N(2)=(1+Zi)*(1-EtA)/4;
N(3)=(1+Zi)*(1+EtA)/4;
N(4)=(1-Zi)*(1+EtA)/4;

dNdz(1)=-(1-EtA)/4;
dNdz(2)=(1-EtA)/4;
dNdz(3)=(1+EtA)/4;
dNdz(4)=-(1+EtA)/4;

dNde(1)=-(1-Zi)/4;
dNde(2)=-(1+Zi)/4;
dNde(3)=(1+Zi)/4;
dNde(4)=(1-Zi)/4;

end


function[A]=quadrature(n)

if(n==1)
    A(1,1:2)=[0 2];
end

if(n==2)
    A(1,1:2)=[-1/sqrt(3) 1];
    A(2,1:2)=[1/sqrt(3) 1];
end

if(n==3)
    A(1,1:2)=[-sqrt(3/5) 5/9];
    A(2,1:2)=[0 8/9];
    A(3,1:2)=[sqrt(3/5) 5/9];
end

end
%% Mesh creation function modified for Quad elements
function[nEl,nNode,EleConnArr,xGlobalC]=createmesh(nElx,nEly,W,H)

nNodeEl=4;

nEl=nElx*nEly;

nNode=(nElx+1)*(nEly+1);

elemW=W/nElx;
elemH=H/nEly;


%%% Global node coordinates

for i=1:nEly+1
    for j=1:nElx+1
        xGlobalC(j+(i-1)*(nElx+1),1)=(j-1)*elemW;
        xGlobalC(j+(i-1)*(nElx+1),2)=(i-1)*elemH;
    end
end


% Connectivity array for Quad elements

for i=1:nEly
    for j=1:nElx
        k=j+(i-1)*(nElx);
        EleConnArr(k,1)=k+(i-1);
        EleConnArr(k,2)=k+(i);
        EleConnArr(k,3)=k+(i)+nElx+1;
        EleConnArr(k,4)=k+(i)+nElx;
        
    end
end

end

function[tDelta]=tDelta_calculate(alpha, mDirichlet, kDirichlet)
if(alpha<0.5)
    e=eig(mDirichlet\kDirichlet);
    e_max=max(e);
    tDelta=2/(1-2*alpha)/e_max;
end

if(alpha>=0.5)
    tDelta=10;
end

end










