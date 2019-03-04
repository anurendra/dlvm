function [Pf_z, Pz_t, d] = dplca( f, K,q, iter, speaker, time_train, sz,hop,Fs, n_alpha, Pf_z, Pz_t, pl, learn_Pf_z, learn_Pz_t)
% function [Pf_z,Pz_t] = plca( x, K, iter, sz, sw, sh, z, w, h, pl, lw, lh)
%
% Perform dynamic PLCA
%
% Inputs:
%  x     input distribution
%  K     number of components
%  iter  number of EM iterations [default = 100]
%  z     initial value of p(z) [default = random]
%  w     initial value of p(w) [default = random]
%  h     initial value of p(h) [default = random]
%  pl    plot flag [default = 1]
%  learn_Pf_z    columns of Pf_z to learn [default = 1:K]
%  learn_Pz_t    columns of Pz_t to learn [default = 1:n_t]
%
% Outputs: 
%  w   p(f_z) - latent bases
%  h   p(z_t) - component bases
%'hu ha ha'
par.result_dir=['Results' '/' 'Results_' num2str(time_train) '_' num2str(q)  ];
addpath('perceptual/bss_eval');
addpath('perceptual');
if(~exist(par.result_dir))
   mkdir(par.result_dir);
   %chmod 777 par.result_dir;
end;

if(n_alpha==1)
	matFileName = [par.result_dir '/' 's' num2str(speaker) '_' num2str(K)  '_' 'dynamic' '.mat'];
else
	matFileName = [par.result_dir '/' 'n' num2str(speaker) '_' num2str(K)  '_' 'dynamic' '.mat'];
end;
x=abs(f);
N=sum(x,1);
[n_f, n_t] = size( x);
%alpha_t=n_alpha*is_dynamic;

N_f=N(1:end-1);
%N_f=N(1:end-1) .* is_dynamic(2:end);  %
if ~exist( 'q')
	q = 1;
end

if ~exist( 'iter')
	iter = 400;
end

% Default plot flag
if ~exist( 'pl')
	pl = 1;
end

% Learn Pf_z by default
if ~exist( 'learn_Pf_z')
	learn_Pf_z = 1:K;
end

% Learn Pz_t by default
if ~exist( 'learn_Pz_t')
	learn_Pz_t = 1:n_t;
end

if ~exist('alpha_t')
    alpha_t=zeros(length(learn_Pz_t),1);
end

% Initialize
if ~exist( 'Pf_z') || isempty(Pf_z)
	if exist(['Initialization_'  num2str(K) '.mat'])
		 temp= load(['Initialization_' num2str(K) '.mat']);
		 Pf_z=temp.Pf_z;
		 Pz_t=temp.Pz_t;
	else
		Pf_z = rand(n_f , K);
	end
elseif size(Pf_z, 2) ~= K
	Pf_z = [Pf_z rand( n_f, K-size( Pf_z, 2))];
end
Pf_z=bsxfun(@rdivide,Pf_z,sum(Pf_z,1));

if ~exist( 'Pz_t') || isempty( Pz_t)
	Pz_t = rand( K, n_t);
elseif size( Pz_t, 1) ~= K
	Pz_t = [Pz_t; rand( K-size( Pz_t, 1), n_t)];
end
Pz_t = bsxfun(@rdivide,Pz_t,sum(Pz_t,1));

save(['Initialization_' num2str(K) '.mat'],'Pf_z','Pz_t');

% Iterate

d=zeros(K,1);
iter2=8;

%depen=zeros(K^2,iter2,iter);
depen=zeros(K,iter2,iter);
%qq=0.2;
d_i=rand(1);
log_l=zeros(iter2,iter);
%	log_l(k,it)=log_likelihood_dd(Pz_t,N_f,d,x,Pf_z, Pz_t);
%	
options = optimset('MaxIter', 100000, 'MaxFunEvals', 100000);
		        		%'hu'
options = optimset('GradObj','on');

if(~exist(matFileName))

	for it = 1:iter
		% E-step
		temp = x ./ (Pf_z*Pz_t);

		% M-step
		if ~isempty( learn_Pf_z)
			nf_z = Pf_z .* (temp*Pz_t');
			Pf_z(:,learn_Pf_z)= bsxfun(@rdivide, nf_z(:,learn_Pf_z) ,sum(nf_z(:,learn_Pf_z) ,1));
	    end
	    
		if ~isempty( learn_Pz_t)
			it  
			%temp = x ./ (Pf_z*Pz_t); 
			if(it<50)
				
				
				nz_t = Pz_t.*(Pf_z'*temp);
				Pz_t= bsxfun(@rdivide, nz_t,sum(nz_t ,1));
				log_l(:,it)=log_likelihood_dd(Pf_z,Pz_t,N_f,d,x,Pf_z, Pz_t)*ones(iter2,1);
			

            else

                
	        	    	
	        	for k=1:iter2                    
	        		%temp = x ./ (Pf_z*Pz_t);
	        		nz_t = Pz_t.*(Pf_z'*temp);

	        		%nz_t(:,1) = Pz_t(:,1).*(Pf_z'*temp(:,1));
					Pz_t(:,1) = bsxfun(@rdivide, nz_t(:,1),sum(nz_t(:,1) ,1));  	        				
	        		
	        		for(t= learn_Pz_t(2:end))
						nz_t(:,t) = nz_t(:,t) + (N(t-1)* d.*Pz_t(:,t-1)).^q;
						Pz_t(:,t)= bsxfun(@rdivide, nz_t(:,t) ,sum(nz_t(:,t) ,1));
		      		end
		      		d= fmincon(@(d)nll_gradient_dd(d,K,Pz_t,N_f) , d,[],[],[],[],zeros(K,1),inf(K,1),[],options);	
		      		%d=gen_newton(d,K,Pz_t,N_f);			        	
				        	%depen(:,k,it)=d;
		      		%log_l(2*k-1,it)=log_likelihood_dd(Pf_z,Pz_t,N_f,d,x,Pf_z, Pz_t);
	        					     		
		 			
				    log_l(k,it)=log_likelihood_dd(Pf_z,Pz_t,N_f,d,x,Pf_z, Pz_t);
                    
					
		        end;
	        end
		end	
		

		
	end

temp1=Pf_z*Pz_t;
sound_r=temp1*diag(N);
fp = f ./abs(f);
%matFileName = [par.result_dir '/' 's' num2str(speaker) '_' num2str(K) '_' num2str(time_train) '_' 'dynamic' '.mat'];
save(matFileName,'Pf_z','Pz_t','f','d','depen','log_l');

y = stft( sound_r.*fp, sz, hop , 0, 'hann');

   %y1=
audiofileName=[par.result_dir  '/' 'audio' num2str(speaker) '_' num2str(K) '_' 'dynamic'  '.wav'];
audiowrite(audiofileName,y,Fs);

end;

%----------------------------------------------------------------------------------------------------------------------------------
function [log_l] = log_likelihood_dd(Pf_z,Pz_t,N_f,d,x, old_Pf_z, old_Pz_t)
 %function [log_l] = log_likelihood_dd(Pz_t,N_f,d,x, old_Pf_z, old_Pz_t)
  state_lag=Pz_t(:,(1:end-1));
  hyper_par=calculate_hyperpar(state_lag, N_f,d);
  log_l_dd = sum(gammaln(sum(hyper_par,1))) - sum(sum(gammaln(hyper_par),1)) + sum(sum((hyper_par-1).*log(Pz_t(:,2:end)+eps)));
  temp = x ./ (old_Pf_z * old_Pz_t);
  %temp = x ./ (Pf_z*Pz_t);
  temp1 =  (old_Pz_t .*log(Pz_t+eps)) .* (old_Pf_z' *temp);
  temp2 =  (old_Pf_z .*log(Pf_z+eps)) .* (temp* old_Pz_t');
  %temp2= Pz_t.*((Pf_z'.* log(Pf_z)')*temp);
  %log_l = sum(sum(temp1)) + sum(sum(temp2)) + log_l_dd;
  log_l = sum(sum(temp1))  + sum(sum(temp2))+ log_l_dd;


%------------------------------------------------------------------------------------------------------------------------------------
function hyper_par = calculate_hyperpar(state_lag,N_f,d)
  %state_lag=Pz_t(:,(1:end-1));
  hyper_par= bsxfun(@times,state_lag ,N_f);
  hyper_par= bsxfun(@times, hyper_par,d);
  hyper_par= hyper_par + 1;
