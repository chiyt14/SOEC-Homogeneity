clear all;
load('surrogate_model_for_upload.mat');
X_MAX=[750,240,300,1.7]; % get the maximum value of each input (each column)
X_MIN=[600,60,40,1.0]; % get the minimum value of each input (each column)
X_BASE=[675,100,170,1.35]; % used in plot the prediction surface of the NNs
rng(1919810);
Ns=100000;
d=4;
% Sample=rand(Ns,d*2);
Sample=lhsdesign(Ns,d*2);
A=Sample(:,1:d);
B=Sample(:,d+1:2*d);
AB=zeros(Ns,d,d);
for i=1:d
    AB(:,:,i)=A;
    AB(:,i,i)=B(:,i);
end
Iup_s_A=NET{1}(A')';
Iup_s_B=NET{1}(B')';
Imid_s_A=NET{2}(A')';
Imid_s_B=NET{2}(B')';
Idown_s_A=NET{3}(A')';
Idown_s_B=NET{3}(B')';
Tmax_s_A=NET{5}(A')';
Tmax_s_B=NET{5}(B')';
Tmin_s_A=NET{6}(A')';
Tmin_s_B=NET{6}(B')';
Qst_s_A=A(:,3)*(X_MAX(3)-X_MIN(3))+X_MIN(3);
Qst_s_B=B(:,3)*(X_MAX(3)-X_MIN(3))+X_MIN(3);
Iup_s_AB=zeros(Ns,d);
Imid_s_AB=zeros(Ns,d);
Idown_s_AB=zeros(Ns,d);
Tmax_s_AB=zeros(Ns,d);
Tmin_s_AB=zeros(Ns,d);
Qst_s_AB=zeros(Ns,d);
for i=1:d
    Iup_s_AB(:,i)=NET{1}(AB(:,:,i)')';
    Imid_s_AB(:,i)=NET{2}(AB(:,:,i)')';
    Idown_s_AB(:,i)=NET{3}(AB(:,:,i)')';
    Tmax_s_AB(:,i)=NET{5}(AB(:,:,i)')';
    Tmin_s_AB(:,i)=NET{6}(AB(:,:,i)')';
    Qst_s_AB(:,i)=AB(:,3,i)*(X_MAX(3)-X_MIN(3))+X_MIN(3);
end
IH_I_A=1-Idown_s_A./Iup_s_A;
IH_I_B=1-Idown_s_B./Iup_s_B;
IH_I_AB=1-Idown_s_AB./Iup_s_AB;
IH_T_A=Tmax_s_A-Tmin_s_A;
IH_T_B=Tmax_s_B-Tmin_s_B;
IH_T_AB=Tmax_s_AB-Tmin_s_AB;
SU_A=(Iup_s_A+Imid_s_A+Idown_s_A)./(Qst_s_A*0.5/60/1000/22.4*2*96485);
SU_B=(Iup_s_B+Imid_s_B+Idown_s_B)./(Qst_s_B*0.5/60/1000/22.4*2*96485);
SU_AB=(Iup_s_AB+Imid_s_AB+Idown_s_AB)./(Qst_s_AB*0.5/60/1000/22.4*2*96485);

SU_s=[SU_A',SU_B',reshape(SU_AB,1,[])];
NUI_s=[IH_I_A',IH_I_B',reshape(IH_I_AB,1,[])];
NUT_s=[IH_T_A',IH_T_B',reshape(IH_T_AB,1,[])];
Var_SU=sum(SU_s.^2)/length(SU_s)-mean(SU_s)^2;
Var_NUI=sum(NUI_s.^2)/length(NUI_s)-mean(NUI_s)^2;
Var_NUT=sum(NUT_s.^2)/length(NUT_s)-mean(NUT_s)^2;
for i=1:d
    S_SU_s(i)=(1/Ns*sum(SU_B.*(SU_AB(:,i)-SU_A)))/Var_SU;
    S_NUI_s(i)=(1/Ns*sum(IH_I_B.*(IH_I_AB(:,i)-IH_I_A)))/Var_NUI;
    S_NUT_s(i)=(1/Ns*sum(IH_T_B.*(IH_T_AB(:,i)-IH_T_A)))/Var_NUT;
    ST_SU_s(i)=(1/Ns/2*sum((SU_AB(:,i)-SU_A).^2))/Var_SU;
    ST_NUI_s(i)=(1/Ns/2*sum((IH_I_AB(:,i)-IH_I_A).^2))/Var_NUI;
    ST_NUT_s(i)=(1/Ns/2*sum((IH_T_AB(:,i)-IH_T_A).^2))/Var_NUT;
end