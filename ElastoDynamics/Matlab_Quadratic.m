%% Variable values
W = 0.1;  % z-direction
H = 0.1;  % y-direction
L = 1;    % x-direction
E = 1000; % youngs modulus
nu = 0.3; % poisson's ratio
rho = 1;  % density
%%% rayleigh damping constants
a = 1;
b = 0.001;
%%% Lame's parameters
lambda = E / 2 / (1 + nu);
mu = E * nu / (1 + nu) / (1 - 2 * nu);
delta = eye(3);
C = zeros(3, 3, 3, 3);

%%% Cijkl Elastic moduli
for i=1:3
    for j=1:3
        for k=1:3
            for l=1:3
                C(i,j,k,l)=lambda*delta(i,j)*delta(k,l)  + 2*mu*(delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k));
            end
        end
    end
end
%%
%%Forcing it to be only quadratic
order = 2;  % Force quadratic elements

%% Dirichlet displacement at z=L
yDisp = 0.05;

%% time stepping parameters
beta = 1/6;
gamma = 1/2;
transient = 1; % (transient =1) for performing transient analysis, otherwise (transient= 0) for static
tDelta = 1000; % deltaT
tSteps = 10;   % number of time steps

%% Gaussian quadrature  %%% select from 1, 2 or 3
wq = 3;  % Use at least 3-point quadrature for quadratic accuracy

%% Meshing
nElx = 4;  %%W
nEly = 4;  %%H
nElz = 40; %%L

nNoEl = 20;  %% 20 nodes in quadratic hex 
nDof = 3;    %% nDof;

[nEl, nNodes, connArr, xGlo] = createmesh(nElx, nEly, nElz, L, W, H, order); 

%% Dirichlet BC 
j = 1;
for i = 1:nNodes
    if (xGlo(i, 3) == 0) % Dirichlet at z=0
        dirich(j, 1:2) = [3 * (i - 1) + 1, 0];
        dirich(j + 1, 1:2) = [3 * (i - 1) + 2, 0];
        dirich(j + 2, 1:2) = [3 * (i - 1) + 3, 0];
        j = j + 3;
    end
    if (xGlo(i, 3) == L) % Dirichlet at z=L
        dirich(j, 1:2) = [3 * (i - 1) + 2, yDisp];
        j = j + 1;
    end
end
nDirich = length(dirich);

%% Initial conditions (No Changes Needed)
u0 = zeros(nDof * nNodes, 1); % initial displacement
v0 = zeros(nDof * nNodes, 1); % initial velocity

%% kLocal and kGlobal 
gaussMatrix = quadrature(wq); % Gaussian quadrature (Ensure wq is suitable for quadratic) 
kGlobal = zeros(nDof * nNodes, nDof * nNodes);
mGlobal = zeros(nDof * nNodes, nDof * nNodes);
fGlobal = zeros(nDof * nNodes, 1);

%% kLocal, mLocal
for e = 1:nEl
    kLocal = zeros(nDof * nNoEl, nDof * nNoEl); % Size adjusted for quadratic
    mLocal = zeros(nDof * nNoEl, nDof * nNoEl); % Size adjusted for quadratic
    fLocal = zeros(nDof * nNoEl, 1);

    for wqX = 1:wq
        for wqY = 1:wq
            for wqZ = 1:wq

               [N,dNdz,dNde,dNdg] = shapeFunc(gaussMatrix(wqX,1),gaussMatrix(wqY,1),gaussMatrix(wqZ,1),order);

               Jac=[dNdz*xGlo(connArr(e,:),1)   dNde*xGlo(connArr(e,:),1)  dNdg*xGlo(connArr(e,:),1)  ;   %%% Jacobian
                    dNdz*xGlo(connArr(e,:),2)   dNde*xGlo(connArr(e,:),2)  dNdg*xGlo(connArr(e,:),2)  ;
                    dNdz*xGlo(connArr(e,:),3)   dNde*xGlo(connArr(e,:),3)  dNdg*xGlo(connArr(e,:),3) ];
                
                dNdX=[dNdz' dNde' dNdg']*inv(Jac);
                
                
                
                for A=1:nNoEl
                    for B=1:nNoEl
                        K=nodalStiffness(dNdX,C,A,B);
                        M=nodalMass(N,rho,delta,A,B);
                        kL(3*(A-1)+1:3*A,3*(B-1)+1:3*B)=K;
                        mL(3*(A-1)+1:3*A,3*(B-1)+1:3*B)=M;
                    end
                end
                
                kLocal=kLocal+kL*det(Jac)*gaussMatrix(wqX,2)*gaussMatrix(wqY,2)*gaussMatrix(wqZ,2);
                mLocal=mLocal + mL*det(Jac)*gaussMatrix(wqX,2)*gaussMatrix(wqY,2)*gaussMatrix(wqZ,2);
                fLocal=fLocal + 0;    %%%% no forcing term is zero in the question
            end
        end
    end

    % Assembly (No changes needed assuming connArr is correct for quadratic)
    for i = 1:nNoEl 
        for j = 1:nNoEl
            I = 3 * (connArr(e, i) - 1) + 1 : 3 * connArr(e, i);
            J = 3 * (connArr(e, j) - 1) + 1 : 3 * connArr(e, j);
            kGlobal(I, J) = kGlobal(I, J) + kLocal(3 * (i - 1) + 1 : 3 * i, 3 * (j - 1) + 1 : 3 * j);
            mGlobal(I, J) = mGlobal(I, J) + mLocal(3 * (i - 1) + 1 : 3 * i, 3 * (j - 1) + 1 : 3 * j);
        end
    end
end

%% kDirichlet and fDirichlet


kDirichlet=zeros(nDof*nNodes-nDirich,nDof*nNodes-nDirich);
mDirichlet=zeros(nDof*nNodes-nDirich,nDof*nNodes-nDirich);
fDirichlet=zeros(nDof*nNodes-nDirich,1);

nonDirichnodes=zeros(nDof*nNodes-nDirich,1);
j=1;

for i=1:nDof*nNodes
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
end  %% seggragating non-Dirichlet nodes

%%% kDirichlet and mDirichlet
for i = 1:length(nonDirichnodes)
    for j = 1:length(nonDirichnodes)
        kDirichlet(i, j) = kGlobal(nonDirichnodes(i), nonDirichnodes(j));
        mDirichlet(i, j) = mGlobal(nonDirichnodes(i), nonDirichnodes(j));
    end
end

f_dash = zeros(length(nonDirichnodes), 1);

%%% fDirichlet
for i = 1:length(nonDirichnodes)
    for j = 1:nDirich
        f_dash(i) = kGlobal(nonDirichnodes(i), dirich(j, 1)) * dirich(j, 2) + f_dash(i);
    end
end
fDirichlet = fGlobal(nonDirichnodes) - f_dash;

%% static analysis
if (transient == 0)
    kDirichlet = sparse(kDirichlet);
    d = kDirichlet \ fDirichlet; 
    dGlobal = zeros(nDof * nNodes, 1);
    dGlobal(dirich(:, 1)) = dirich(:, 2);
    dGlobal(nonDirichnodes) = d; 

    %%% Plotting for a quadratic mesh  
    typical = (nElx + 1) * (nEly + 1) * (nElz + 1); % All nodes in a quadratic mesh
    dPlot = dGlobal(1 : nDof * typical);

    for i = 1 : typical
        x(i) = dPlot((i - 1) * nDof + 1);
        y(i) = dPlot((i - 1) * nDof + 2);
        z(i) = dPlot((i - 1) * nDof + 3);
    end

    u = [x', y', z'];
    fem_to_vtk_Vector ('HW5', xGlo(1:typical,:), connArr(:,1:8), u); % Assuming connArr is correct for quadratic mesh
end

%% transient analysis

if(transient==1)
    
    
    cDirichlet=(a*mDirichlet + b*kDirichlet);    %%% Rayleigh damping
    
    %%% Initializing
    dGlobal(:,1)=u0;
    vGlobal(:,1)=v0;
    
    %%% time stepping
    
    d(:,1)=dGlobal(nonDirichnodes);
    v(:,1)=vGlobal(nonDirichnodes);
    acc(:,1)=mDirichlet\(fDirichlet - kDirichlet*d(:,1) - cDirichlet*v(:,1) );
    
    for i=1:tSteps
        d_tilde= d(:,i) + tDelta*v(:,i) + (1-2*beta)/2*tDelta^2*acc(:,i);
        v_tilde= v(:,i) + (1-gamma)*tDelta*acc(:,i);
        
        acc(:,i+1)= (mDirichlet+gamma*tDelta*cDirichlet+beta*tDelta^2*kDirichlet)\(fDirichlet - kDirichlet*d_tilde - cDirichlet*v_tilde);
        
        d(:,i+1)= d_tilde + beta*tDelta^2*acc(:,i+1);
        v(:,i+1)= v_tilde + gamma*tDelta*acc(:,i+1);
    end
    
    
    %%% plotting steps
    
    tStepPlot=11;    %%% timestep to plot (should be less than or equal to tSteps+1)
    dGlobal(dirich(:,1))=dirich(:,2);
    dGlobal(nonDirichnodes)=d(:,tStepPlot);
    
    for i=1:nNodes
        x(i)=dGlobal((i-1)*nDof+1);
        y(i)=dGlobal((i-1)*nDof+2);
        z(i)=dGlobal((i-1)*nDof+3);
    end
    
    u=[x',  y', z'];
    
    fem_to_vtk_Vector ('HW5', xGlo, connArr, u);
end

function[nEl,nNodes,connArr,xGlo]=createmesh(nElx,nEly,nElz,L,W,H,~)

order=2;
    
    nEl = nElx*nEly*nElz;
    
    elemL=L/nElz;
    elemH=H/nEly;
    elemW=W/nElx;
    
    %%% Global node coordinates
    for i=1:nElz+1
        for j=1:nEly+1
            for k=1:nElx+1
                xGlo(k+(j-1)*(nElx+1)+(i-1)*(nElx+1)*(nEly+1),1)=(k-1)*elemW; %x-coordinate
                xGlo(k+(j-1)*(nElx+1)+(i-1)*(nElx+1)*(nEly+1),2)=(j-1)*elemH;  %y-coordinate
                xGlo(k+(j-1)*(nElx+1)+(i-1)*(nElx+1)*(nEly+1),3)=(i-1)*elemL;   %z-coordinate
                
            end
        end
    end
    
    typical= (nElx+1)*(nEly+1)*(nElz+1);
    count=1;
    
    for k=1:2*nElz+1
        if(rem(k,2)==1)
            
            for i=1:2*nEly+1
                if(rem(i,2)==1)
                    for j=1:nElx
                        xGlo(typical+count,1)=(2*j-1)*elemW/2;
                        xGlo(typical+count,2)=(i-1)*elemH/2;
                        xGlo(typical+count,3)=(k-1)*elemL/2;
                        count=count+1;
                    end
                end
                if(rem(i,2)==0)
                    for j=1:nElx+1
                        xGlo(typical+count,1)=(j-1)*elemW;
                        xGlo(typical+count,2)=(i-1)*elemH/2;
                        xGlo(typical+count,3)= (k-1)*elemL/2;
                        count=count+1;
                    end
                end
            end
        end
        
        if(rem(k,2)==0)
            
            for i=1:nEly+1
                for j=1:nElx+1
                    xGlo(typical+count,1)=(j-1)*elemW;
                    xGlo(typical+count,2)=(i-1)*elemH;
                    xGlo(typical+count,3)=(k-1)*elemL/2;
                    count=count+1;
                end
            end
            
        end
        
    end
    
    nNodes=length(xGlo);
    
    
    %%% Connectivity array
    connArr=zeros(nEl,20);
    
    for i=1:nElz
        for j=1:nEly
            for k=1:nElx
                l=k+(j-1)*nElx+(i-1)*nElx*nEly;
                
                connArr(l,1)=l+(j-1)+(i-1)*(nElx+nEly+1);
                connArr(l,2)=l+(j)+(i-1)*(nElx+nEly+1);
                connArr(l,3)=l+(j)+nElx+1+(i-1)*(nElx+nEly+1);
                connArr(l,4)=l+(j)+nElx+(i-1)*(nElx+nEly+1);
                
                connArr(l,5)=l+(j-1)+(i)*(nElx+nEly+1)+nElx*nEly;
                connArr(l,6)=l+(j)+(i)*(nElx+nEly+1)+nElx*nEly;
                connArr(l,7)=l+(j)+nElx+1+(i)*(nElx+nEly+1)+nElx*nEly;
                connArr(l,8)=l+(j)+nElx+(i)*(nElx+nEly+1)+nElx*nEly;
                
                connArr(l,9)= typical  + (k)+ (j-1)*(2*nElx+1) + (i-1)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,10)=typical + (nElx+k+1) + (j-1)*(2*nElx+1) + (i-1)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,11)=typical + (2*nElx+1+k) + (j-1)*(2*nElx+1) + (i-1)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,12)=typical + (nElx+k) + (j-1)*(2*nElx+1) + (i-1)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                
                connArr(l,13)= typical  + (k)+ (j-1)*(2*nElx+1) + (i)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,14)=typical + (nElx+k+1) + (j-1)*(2*nElx+1) + (i)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,15)=typical + (2*nElx+1+k) + (j-1)*(2*nElx+1) + (i)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,16)=typical + (nElx+k) + (j-1)*(2*nElx+1) + (i)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                
                connArr(l,17)= typical  + ((nElx+1)*nEly + (nEly+1)*nElx + k)+ (j-1)*(2*nElx+1) + (i-1)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,18)= typical  + ((nElx+1)*nEly + (nEly+1)*nElx + k+1)+ (j-1)*(2*nElx+1) + (i-1)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,19)= typical  + ((nElx+1)*nEly + (nEly+1)*nElx + k + nElx+2)+ (j-1)*(2*nElx+1) + (i-1)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
                connArr(l,20)= typical  + ((nElx+1)*nEly + (nEly+1)*nElx + k + nElx+1)+ (j-1)*(2*nElx+1) + (i-1)*((nElx+1)*(2*nEly+1)+nElx*(nEly+1));
            end
        end
    end
    
end

function [N, dNdz,dNde,dNdg]=shapeFunc(zhi,eta,gamma,~)
xi=zhi;
    zeta=gamma;
    
    N(1)=-(1-xi)*(1-eta)*(1-zeta)*(2+xi+eta+zeta);
    N(2)=-(1+xi)*(1-eta)*(1-zeta)*(2-xi+eta+zeta);
    N(3)=-(1+xi)*(1+eta)*(1-zeta)*(2-xi-eta+zeta);
    N(4)=-(1-xi)*(1+eta)*(1-zeta)*(2+xi-eta+zeta);
    N(5)=-(1-xi)*(1-eta)*(1+zeta)*(2+xi+eta-zeta);
    N(6)=-(1+xi)*(1-eta)*(1+zeta)*(2-xi+eta-zeta);
    N(7)=-(1+xi)*(1+eta)*(1+zeta)*(2-xi-eta-zeta);
    N(8)=-(1-xi)*(1+eta)*(1+zeta)*(2+xi-eta-zeta);
    N(9)=(1-xi*xi)*(1-eta)*(1-zeta)*2;
    N(10)=(1-eta*eta)*(1+xi)*(1-zeta)*2;
    N(11)=(1-xi*xi)*(1+eta)*(1-zeta)*2;
    N(12)=2*(1-eta*eta)*(1-xi)*(1-zeta);
    N(13)=2*(1-xi*xi)*(1-eta)*(1+zeta);
    N(14)=2*(1-eta*eta)*(1+xi)*(1+zeta);
    N(15)=2*(1-xi*xi)*(1+eta)*(1+zeta);
    N(16)=2*(1-eta*eta)*(1-xi)*(1+zeta);
    N(17)=2*(1-zeta*zeta)*(1-xi)*(1-eta);
    N(18)=2*(1-zeta*zeta)*(1+ xi)*(1-eta);
    N(19)=2*(1-zeta*zeta)*(1+xi)*(1+eta);
    N(20)=2*(1-zeta*zeta)*(1- xi)*(1+eta);
    
    N=N/8;
    
    dNdz=[ ((eta - 1)*(zeta - 1)*(eta + 2*xi + zeta + 1))/8,-((eta - 1)*(zeta - 1)*(eta - 2*xi + zeta + 1))/8, -((eta + 1)*(zeta - 1)*(eta + 2*xi - zeta - 1))/8, -((eta + 1)*(zeta - 1)*(2*xi - eta + zeta + 1))/8, -((eta - 1)*(zeta + 1)*(eta + 2*xi - zeta + 1))/8, ((eta - 1)*(zeta + 1)*(eta - 2*xi - zeta + 1))/8, ((eta + 1)*(zeta + 1)*(eta + 2*xi + zeta - 1))/8, -((eta + 1)*(zeta + 1)*(eta - 2*xi + zeta - 1))/8, -(xi*(eta - 1)*(zeta - 1))/2, ((eta^2 - 1)*(zeta - 1))/4,(xi*(eta + 1)*(zeta - 1))/2,-((2*eta^2 - 2)*(zeta - 1))/8,(xi*(eta - 1)*(zeta + 1))/2,-((2*eta^2 - 2)*(zeta + 1))/8,-(xi*(eta + 1)*(zeta + 1))/2,((2*eta^2 - 2)*(zeta + 1))/8, -((2*zeta^2 - 2)*(eta - 1))/8,((2*zeta^2 - 2)*(eta - 1))/8,-((2*zeta^2 - 2)*(eta + 1))/8,((2*zeta^2 - 2)*(eta + 1))/8];

    dNde=[((xi - 1)*(zeta - 1)*(2*eta + xi + zeta + 1))/8, -((xi + 1)*(zeta - 1)*(2*eta - xi + zeta + 1))/8, -((xi + 1)*(zeta - 1)*(2*eta + xi - zeta - 1))/8, -((xi - 1)*(zeta - 1)*(xi - 2*eta + zeta + 1))/8,-((xi - 1)*(zeta + 1)*(2*eta + xi - zeta + 1))/8,((xi + 1)*(zeta + 1)*(2*eta - xi - zeta + 1))/8,((xi + 1)*(zeta + 1)*(2*eta + xi + zeta - 1))/8,-((xi - 1)*(zeta + 1)*(2*eta - xi + zeta - 1))/8, -((xi^2 - 1)*(zeta - 1))/4, (eta*(xi + 1)*(zeta - 1))/2, ((xi^2 - 1)*(zeta - 1))/4,  -(eta*(xi - 1)*(zeta - 1))/2, ((2*xi^2 - 2)*(zeta + 1))/8,  -(eta*(xi + 1)*(zeta + 1))/2,  -((2*xi^2 - 2)*(zeta + 1))/8,  (eta*(xi - 1)*(zeta + 1))/2, -((2*zeta^2 - 2)*(xi - 1))/8,  ((2*zeta^2 - 2)*(xi + 1))/8, -((2*zeta^2 - 2)*(xi + 1))/8,  ((2*zeta^2 - 2)*(xi - 1))/8];

    dNdg=[((eta - 1)*(xi - 1)*(eta + xi + 2*zeta + 1))/8, -((eta - 1)*(xi + 1)*(eta - xi + 2*zeta + 1))/8, -((eta + 1)*(xi + 1)*(eta + xi - 2*zeta - 1))/8, -((eta + 1)*(xi - 1)*(xi - eta + 2*zeta + 1))/8, -((eta - 1)*(xi - 1)*(eta + xi - 2*zeta + 1))/8, ((eta - 1)*(xi + 1)*(eta - xi - 2*zeta + 1))/8, ((eta + 1)*(xi + 1)*(eta + xi + 2*zeta - 1))/8, -((eta + 1)*(xi - 1)*(eta - xi + 2*zeta - 1))/8, -((xi^2 - 1)*(eta - 1))/4, ((eta^2 - 1)*(xi + 1))/4, ((xi^2 - 1)*(eta + 1))/4,  -((2*eta^2 - 2)*(xi - 1))/8, ((2*xi^2 - 2)*(eta - 1))/8, -((2*eta^2 - 2)*(xi + 1))/8, -((2*xi^2 - 2)*(eta + 1))/8, ((2*eta^2 - 2)*(xi - 1))/8,  -(zeta*(eta - 1)*(xi - 1))/2, (zeta*(eta - 1)*(xi + 1))/2, -(zeta*(eta + 1)*(xi + 1))/2, (zeta*(eta + 1)*(xi - 1))/2];
    
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

function[K]=nodalStiffness(dNdX,C,A,B)

K=zeros(3,3);
dNdX_t=dNdX';

for i=1:3
    for j=1:3
        for k=1:3
            for l=1:3
                K(i,k)= K(i,k)+ dNdX(A,j)*C(i,j,k,l)*dNdX_t(l,B);
            end
        end
    end
end
end

function [M]=nodalMass(N,rho,delta,A,B)
M=zeros(3,3);
for i=1:3
    for k=1:3
        M(i,k)=N(A)*rho*delta(i,k)*N(B);
    end
end

end