function [X1, X2] = source_sep(X_mix,Pf_z1,Pf_z2,K)
X=abs(X_mix);
phi= X_mix ./X;

n_f  = size(X,1); % vocabulary size
n_t = size(X,2); % # of documents


% allocate memory for the posterior
Pz1_ft = zeros(K,n_f,n_t); 
Pz2_ft = zeros(K,n_f,n_t);
Li    = [];
%maxit = Par.maxit;-
maxit=200;
Lik=zeros(maxit,1);
[P1_t, P2_t, Pz_1t, Pz_2t] = source_sep_init(2, n_f,n_t,K);
% EM algorithm
for it = 1:maxit   
   [Pz1_ft, Pz2_ft] = source_sep_Estep(P1_t, P2_t, Pf_z1, Pf_z2, Pz_1t, Pz_2t);
   
   % M-step
   [P1_t, P2_t, Pz_1t, Pz_2t] = source_sep_Mstep(X,Pz1_ft, Pz2_ft);
end;
Pf_1t=Pf_z1*Pz_1t;
Pf_2t=Pf_z2*Pz_2t;
temp1= Pf_1t* diag(P1_t);
temp2 = Pf_2t * diag(P2_t);
X1_mag= temp1./(temp1+temp2);
X2_mag= temp2./(temp1+temp2);
X1_mag=X1_mag.*X;
X2_mag = X2_mag.*X;
X1 = X1_mag.*phi;
X2 = X2_mag.*phi;
%y1 = stft(X1, sz, hop , 0, 'hann');
%y2 = stft(X2, sz, hop , 0, 'hann');
%%soundsc(y1,16000);
%soundsc(y2,16000);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize conditional probabilities for EM 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P1_t, P2_t, Pz_1t, Pz_2t] = source_sep_init(n_s, n_f,n_t,K)
% n_f  ... number of frequencies
% n_t ... number of time instants
% K  ... number of bases
%
% Pf_z ... P(f|z)
% Pz_t ... P(z) at time t

 P1_t=0.5*ones(n_t,1);
% P1_t=bsxfun(@rdivide,P1_t,sum(P1_t,1));
 P2_t=P1_t;
 
% random assignment
%Pf_z =rand(n_f,K);   % word probabilities conditioned on topic
%Pf_z=bsxfun(@rdivide,Pf_z,sum(Pf_z,1));
% random assignment
Pz_1t = rand(K,n_t);
Pz_1t = bsxfun(@rdivide,Pz_1t,sum(Pz_1t,1));
Pz_2t = Pz_1t;
%Pz_ft=rand(K,n_f,n_t);

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) E step compute posterior on z,  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Pz1_ft, Pz2_ft] = source_sep_Estep(P1_t, P2_t, Pf_z1, Pf_z2, Pz_1t, Pz_2t)
   %K = size(Pw_z,2);

   [K,n_t]=size(Pz_1t);
   n_f=size(Pf_z1,1);
   Pz1_ft=zeros(K,n_f,n_t);
   Pz2_ft = Pz1_ft;
   %n_t=size(Pz_t,2);
   for t=1:n_t
      temp1=P1_t(t)*Pf_z1*diag(Pz_1t(:,t));
     
      temp2=P2_t(t)*Pf_z2*diag(Pz_2t(:,t));
      
      %Pz1_ft(:,:,t)=
      %Pz2_ft(:,:,t)=bsxfun(@rdivide,temp',sum(temp',1));
      Pz1_ft(:,:,t)=bsxfun(@rdivide,temp1',sum(temp1',1)+sum(temp2',1));
      Pz2_ft(:,:,t)=bsxfun(@rdivide,temp2',sum(temp1',1)+sum(temp2',1));
      %norma=Pz1_ft(:,:,t) + Pz2_ft(:,:,t);
      %Pz1_ft(:, :, t) = Pz1_ft(:,:,t)./ norma;
      %Pz2_ft(:,:,t) = Pz2_ft(:,:,t) ./ norma;
   end


   
return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  (2) M step, maximize log-likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [P1_t, P2_t, Pz_1t, Pz_2t] = source_sep_Mstep(X,Pz1_ft, Pz2_ft)
   %K = size(Pz_dw,3);
   [K,n_f,n_t]=size(Pz1_ft);
   Pz_1t=zeros(K,n_t);
   P1_t=zeros(n_t,1);
   P2_t=P1_t;
   
   
   for t=1:n_t
     % size(Pz1_ft)
     % size(X(:,t))
      temp=Pz1_ft(:,:,t)*X(:,t);
      Pz_1t(:,t)=bsxfun(@rdivide,temp,sum(temp));
      temp=Pz2_ft(:,:,t)*X(:,t);
      Pz_2t(:,t)=bsxfun(@rdivide,temp,sum(temp));
   end

   for t=1:n_t
      temp=Pz1_ft(:,:,t)*X(:,t);
      temp1 = sum(temp,1);
      temp=Pz2_ft(:,:,t)*X(:,t);
      temp2 = sum(temp,1);   
      P1_t(t) = temp1/(temp1+temp2);
      P2_t(t) = temp2/(temp1 + temp2);  
   end
   

   
 
return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data log-likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = dynamic_plca_logL(X,Pf_z,Pz)
   L = sum(sum(X .* log(Pf_z * Pz + eps)));
return;



return;
