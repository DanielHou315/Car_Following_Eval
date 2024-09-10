% Task 7 Time headway during traffic jams

load('SQLprogram/Task789Data.mat');     % Jam
load('SQLprogram/Task789LabeledBrakeEvent.mat');    % CandiTrip
load('SQLprogram/Task789ValidBrakeEvent.mat');      % QueryData

% QueryData header:
% 1=driver; 2=trip; 3=1st brake start; 4=last brake end; 5=jam id;
% 6=1st brake road type; 7=last brake road type

% Jam header:
% 1=driver; 2=trip; 3=time; 4=speed; 5=accelpedal; 6=Ax; 7=range;
% 8=rangerate; 9=TTC; 10=targetid; 11=traffic count; 12=age; 13=gender

% CandiTrip is a table with its own headers

% Create Time-headway for Jam
Jam(:,14) = Jam(:,7)./Jam(:,4);     % now 14=time headway
Jam(isinf(Jam(:,14)),14)=NaN;
%% quick view for 160 congestions
PerJam = [];
for i=1:size(QueryData,1)
    MeanRangeInAJam = mean(Jam(Jam(:,1)==QueryData(i,1) & Jam(:,2)==QueryData(i,2) & Jam(:,3)>=QueryData(i,3) & Jam(:,3)<=QueryData(i,4),7));
    MeanSpeedInAJam = mean(Jam(Jam(:,1)==QueryData(i,1) & Jam(:,2)==QueryData(i,2) & Jam(:,3)>=QueryData(i,3) & Jam(:,3)<=QueryData(i,4),4));
    MeanTHInAJam = mean(Jam(Jam(:,1)==QueryData(i,1) & Jam(:,2)==QueryData(i,2) & Jam(:,3)>=QueryData(i,3) & Jam(:,3)<=QueryData(i,4),14),'omitnan');
    BrakeStartRoadType = CandiTrip{CandiTrip{:,1}==QueryData(i,1) & CandiTrip{:,2}==QueryData(i,2) & CandiTrip{:,5}==QueryData(i,3),9};
    PerJam = [PerJam; MeanRangeInAJam, MeanSpeedInAJam, MeanTHInAJam, BrakeStartRoadType];
end
%% at full stop
BrakeToFullStop = CandiTrip{CandiTrip{:,20}~=0 & CandiTrip{:,15}==0, [1 2 6 9]};
AtStop=[];
for j=1:size(BrakeToFullStop,1)
    AtStop = [AtStop; BrakeToFullStop(j,[1 2 4]), Jam(Jam(:,1)==BrakeToFullStop(j,1) & Jam(:,2)==BrakeToFullStop(j,2) & Jam(:,3)==BrakeToFullStop(j,3), [7 9 11 12 13])];
end
% AtStop header:
% 1=driver; 2=trip; 3=road type; 4=range; 5=TTC; 6=traffic count; 7=age; 8=gender

%% plot - summary, mean speed
figure
histogram(PerJam(PerJam(:,4)==1,2)*2.23694,[0:5:30])
title('Freeways');
xlabel('Mean speed (MPH)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==3,2)*2.23694,[0:5:30])
title('Major surface roads');
xlabel('Mean speed (MPH)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==4,2)*2.23694,[0:5:30])
title('Minor surface roads');
xlabel('Mean speed (MPH)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==5,2)*2.23694,[0:5:30])
title('Local roads');
xlabel('Mean speed (MPH)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==6,2)*2.23694,[0:5:30])
title('Ramps');
xlabel('Mean speed (MPH)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

%% plot - summary, mean range
figure
histogram(PerJam(PerJam(:,4)==1,1),[0:5:40])
title('Freeways');
xlabel('Mean distance to the lead vehicle (m)');
ylabel('Frequency');
xlim([0 40])
%ylim([0 200])
xticks([0:5:40])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==3,1),[0:5:40])
title('Major surface roads');
xlabel('Mean distance to the lead vehicle (m)');
ylabel('Frequency');
xlim([0 40])
%ylim([0 200])
xticks([0:5:40])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==4,1),[0:5:40])
title('Minor surface roads');
xlabel('Mean distance to the lead vehicle (m)');
ylabel('Frequency');
xlim([0 40])
%ylim([0 200])
xticks([0:5:40])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==5,1),[0:5:40])
title('Local roads');
xlabel('Mean distance to the lead vehicle (m)');
ylabel('Frequency');
xlim([0 40])
%ylim([0 200])
xticks([0:5:40])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==6,1),[0:5:40])
title('Ramps');
xlabel('Mean distance to the lead vehicle (m)');
ylabel('Frequency');
xlim([0 40])
%ylim([0 200])
xticks([0:5:40])
v=gca;
v.FontSize=16;

%% plot - summary, mean time-headway
figure
histogram(PerJam(PerJam(:,4)==1,3),[0:2:20])
title('Freeways');
xlabel('Mean time headway to the lead vehicle (s)');
ylabel('Frequency');
xlim([0 20])
%ylim([0 200])
xticks([0:5:20])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==3,3),[0:2:20])
title('Major surface roads');
xlabel('Mean time headway to the lead vehicle (s)');
ylabel('Frequency');
xlim([0 20])
%ylim([0 200])
xticks([0:5:20])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==4,3),[0:2:20])
title('Minor surface roads');
xlabel('Mean time headway to the lead vehicle (s)');
ylabel('Frequency');
xlim([0 20])
%ylim([0 200])
xticks([0:5:20])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==5,3),[0:2:20])
title('Local roads');
xlabel('Mean time headway to the lead vehicle (s)');
ylabel('Frequency');
xlim([0 20])
%ylim([0 200])
xticks([0:5:20])
v=gca;
v.FontSize=16;

figure
histogram(PerJam(PerJam(:,4)==6,3),[0:2:20])
title('Ramps');
xlabel('Mean time headway to the lead vehicle (s)');
ylabel('Frequency');
xlim([0 20])
%ylim([0 200])
xticks([0:5:20])
v=gca;
v.FontSize=16;

%% plot - summary, association scatters
figure
scatter(PerJam(PerJam(:,4)==1,2)*2.23694, PerJam(PerJam(:,4)==1,1))
title('Freeways');
xlabel('Mean speed (s)');
ylabel('Distance (m)');
xlim([0 30])
ylim([0 40])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
scatter(PerJam(PerJam(:,4)==3,2)*2.23694, PerJam(PerJam(:,4)==3,1))
title('Major surface roads');
xlabel('Mean speed (s)');
ylabel('Distance (m)');
xlim([0 30])
ylim([0 40])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
scatter(PerJam(PerJam(:,4)==4,2)*2.23694, PerJam(PerJam(:,4)==4,1))
title('Minor surface roads');
xlabel('Mean speed (s)');
ylabel('Distance (m)');
xlim([0 30])
ylim([0 40])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
scatter(PerJam(PerJam(:,4)==5,2)*2.23694, PerJam(PerJam(:,4)==5,1))
title('Local roads');
xlabel('Mean speed (s)');
ylabel('Distance (m)');
xlim([0 30])
ylim([0 40])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
scatter(PerJam(PerJam(:,4)==6,2)*2.23694, PerJam(PerJam(:,4)==6,1))
title('Ramps');
xlabel('Mean speed (s)');
ylabel('Distance (m)');
xlim([0 30])
ylim([0 40])
xticks([0:5:30])
v=gca;
v.FontSize=16;



%% plotting - at complete stop, road type
figure
histogram(AtStop(AtStop(:,3)==1,4),[0:2:30])
title('Freeways');
xlabel('Distance to the lead vehicle at full stop (m)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(AtStop(AtStop(:,3)==3,4),[0:2:30])
title('Major surface roads');
xlabel('Distance to the lead vehicle at full stop (m)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(AtStop(AtStop(:,3)==4,4),[0:2:30])
title('Minor surface roads');
xlabel('Distance to the lead vehicle at full stop (m)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(AtStop(AtStop(:,3)==5,4),[0:2:30])
title('Local roads');
xlabel('Distance to the lead vehicle at full stop (m)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(AtStop(AtStop(:,3)==6,4),[0:2:30])
title('Ramps');
xlabel('Distance to the lead vehicle at full stop (m)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

%% age
figure
histogram(AtStop(AtStop(:,7)==1,4),[0:2:30])
title('Driver of 20-30 yr');
xlabel('Distance to the lead vehicle at full stop (m)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(AtStop(AtStop(:,7)==2,4),[0:2:30])
title('Driver of 40-50 yr');
xlabel('Distance to the lead vehicle at full stop (m)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

figure
histogram(AtStop(AtStop(:,7)==3,4),[0:2:30])
title('Driver of 60-70 yr');
xlabel('Distance to the lead vehicle at full stop (m)');
ylabel('Frequency');
xlim([0 30])
%ylim([0 200])
xticks([0:5:30])
v=gca;
v.FontSize=16;

%% save csv files
Jam160 = array2table(PerJam,'VariableNames',{'Mean_distance','Mean_speed','mean_time_headway','Road_type'});
AtStopTable = array2table(AtStop(:,[3 4 5 7]),'VariableNames',{'Road_type','Distance','TTC','Age'});

