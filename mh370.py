'''
Created on Mar 27, 2014
'''

'''
Based on:

http://nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold.ipynb

Modeling the Last Flight of MH370 with a Markov Chain Monte Carlo Method
Conor L. Myhrvold

Harvard University, SEAS, IACS , Computational Science & Engineering (CSE)

AM207 : Advanced Scientific Computing: Stochastic Optimization Methods. Monte Carlo Methods for Inference and Data Analysis

Final Project, March 2014

Contact Info:
conor.myhrvold@gmail.com
Twitter: @conormyhrvold
LinkedIn: www.linkedin.com/pub/conor-myhrvold/37/583/a7/ 
'''

#using Anaconda 64 bit distro of Python 2.7, Windows 7 environment (other specs will work though)
#import libraries

#numpy
import numpy as np

#scipy
import scipy as sp
from scipy import stats
from scipy.stats import norm
from scipy.stats import expon

#plotting
#%matplotlib inline 
# ^^ make plots appear in IPython notebook instead of separate window
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from mpl_toolkits.basemap import Basemap # use for plotting lat lon & world maps

################################################################################################################################################################################################

#f = open('runways' + '.txt', 'r') #renamed XPlane data file -- massive list of runway properties around the world

fn = os.path.join(os.path.dirname(__file__), 'runways.txt')
f = open(fn, 'r') #renamed XPlane data file -- massive list of runway properties around the world

line_list = []
for line in f.readlines():
    if '100   ' in line:
        line.rstrip('\n')
        line_list.append(line)
        
################################################################################################################################################################################################
 
print "number of runways:", len(line_list)

################################################################################################################################################################################################

print line_list[0]

################################################################################################################################################################################################

#example of 1 that won't work ... need to filter more
print line_list[76]

################################################################################################################################################################################################

new_line_list = []
for line in line_list:
    if int(line[0]) == 1 and int(line[1]) == 0 and int(line[2]) == 0:
        new_line_list.append(line)

print "number of runways:", len(new_line_list)

################################################################################################################################################################################################

coord1 = []
coord2 = []
coord3 = []
coord4 = []
for line in new_line_list:
    temp = line.split()
    try:
        coord1.append(temp[9])
        coord2.append(temp[10])
        coord3.append(temp[18])
        coord4.append(temp[19])
    except:
        break
    
################################################################################################################################################################################################
    
#check to make sure all lengths equal

#print coord1[0:10]
print len(coord1)
print len(coord2)
print len(coord3)
print len(coord4)

################################################################################################################################################################################################

print coord1[0:10]
print coord2[0:10]
print coord3[0:10]
print coord4[0:10]

################################################################################################################################################################################################

coordlist = np.zeros((len(new_line_list), 4))
for i in xrange(len(coord1)):
    coordlist[i][0] = coord1[i]
    coordlist[i][1] = coord2[i]
    coordlist[i][2] = coord3[i]
    coordlist[i][3] = coord4[i]
#print len(coordlist)
print coordlist[0:10]

################################################################################################################################################################################################

print coordlist[0]

################################################################################################################################################################################################

print coordlist[1][0]+coordlist[1][2], coordlist[1][1]+coordlist[1][3]
print coordlist[2][0]+coordlist[2][2], coordlist[2][1]+coordlist[2][3]

#probably are same runways, just in a criss cross format!

################################################################################################################################################################################################

lat = []
lon = []
for i in xrange(len(coordlist)):
    lat.append( (coordlist[i][0]+coordlist[i][2])/2.0 )
    lon.append( (coordlist[i][1]+coordlist[i][3])/2.0 )
    
print lat[0:10]; print lon[0:10]

################################################################################################################################################################################################

coordlist_average = np.zeros((len(coordlist), 2))

for i in xrange(len(lon)):
    coordlist_average[i][0] = lat[i]
    coordlist_average[i][1] = lon[i]

print coordlist_average[0:10]
print len(coordlist_average) #check to make sure they're all still there!

################################################################################################################################################################################################

earth_radius = 6371000 #in m, http://en.wikipedia.org/wiki/Earth

"""
Haversine equation.
Computes the great circle distance between two pairs of longitude/latitude.
Returns the distance in m or km depending on input (I use meters.) 
"""
def haversine(r,lat1,lon1,lat2,lon2):
    d = 2.0*r*np.arcsin(np.sqrt(np.sin(np.radians(lat2-lat1)/2.0)**2+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(np.radians(lon2-lon1)/2.0)**2))
    return d   

################################################################################################################################################################################################

#testing first data set result
haversine(earth_radius,coordlist[0][0],coordlist[0][1],coordlist[0][2],coordlist[0][3])

################################################################################################################################################################################################

coordlist_len = np.zeros((len(coordlist),1))

for i in xrange(len(coordlist_len)):
    coordlist_len[i] = haversine(earth_radius,coordlist[i][0],coordlist[i][1],coordlist[i][2],coordlist[i][3])
    
print coordlist_len[0:10]

################################################################################################################################################################################################

#convert ft to m
min_run_dist = 5000* 0.3048 # 5000 ft *  0.3048 ft/m
print "The minimum runway distance to land a Boeing 777 is", min_run_dist, "meters."

################################################################################################################################################################################################

runway777_indices = []
for i in xrange(len(coordlist_len)):
    if coordlist_len[i] >= 1524.0 :
        runway777_indices.append(i)
print len(runway777_indices)
print runway777_indices[0:50]

################################################################################################################################################################################################

master_777_runway_coords = coordlist[runway777_indices]
master_777_runway_avg = coordlist_average[runway777_indices]

print len(master_777_runway_coords) == len(master_777_runway_avg)
print "-----------------------"
print master_777_runway_coords
print "-----------------------"
print master_777_runway_avg

################################################################################################################################################################################################

# The Inmarsat satellite is at 0,64.5 -- it's geostationary.
inmarsat_coord = [0, 64.5]

#Now we plot the plane's known positions

#Kuala Lumpur International Airport Coordinates: http://www.distancesfrom.com/my/Kuala-Lumpur-Airport-(KUL)-Malaysia-latitude-longitude-Kuala-Lumpur-Airport-(KUL)-Malaysia-latitude-/LatLongHistory/3308940.aspx
kualalumpur_coord = [2.7544829, 101.7011363]

#Pulau Perak coordinates: http://tools.wmflabs.org/geohack/geohack.php?pagename=Pulau_Perak&params=5_40_50_N_98_56_27_E_type:isle_region:MY
# http://en.wikipedia.org/wiki/Perak_Island -> Indonesia military radar detected near island
pulauperak_coord = [5.680556,98.940833]

# Igari Waypoint. Source: # http://www.fallingrain.com/waypoint/SN/IGARI.html Given in hours,minutes,seconds.
igariwaypoint_coord = [6. + 56./60. + 12./3600., 103. + 35./60. + 6./3600.] 

print "inmarsat lat/lon:", inmarsat_coord[0], inmarsat_coord[1]
print "kuala lumpur int'l airport coord lat/lon:", kualalumpur_coord[0],kualalumpur_coord[1]
print "pulua perak lat/lon:", pulauperak_coord[0],pulauperak_coord[1]
print "igari waypoint lat/lon:", igariwaypoint_coord[0],igariwaypoint_coord[1]

################################################################################################################################################################################################

master_777_runway_avg

################################################################################################################################################################################################

runway_lats = []
runway_lons = []
# split coordinates into list form now to follow plotting example
for i in xrange(len(master_777_runway_avg)):
    runway_lats.append(master_777_runway_avg[i][0])
    runway_lons.append(master_777_runway_avg[i][1])
   
################################################################################################################################################################################################ 
    
#print type(runway_lats); print type(runway_lons)
#print runway_lats[0:10]
#print runway_lons[0:10]

################################################################################################################################################################################################ 

#we similarly split up the plane & sat lat & lons

plane_lats = [2.7544829,(6.+56./60.+12./3600.),5.680556]
plane_lons = [101.7011363,(103.+35./60.+6./3600.),98.940833]

sat_lat = 0
sat_lon = 64.5

################################################################################################################################################################################################ 

#Used Basemap template. Could substitute for any other number of projections, and provide specs for the lat/long displays etc.

# set figure size
fig = plt.figure(figsize=[21,15])

# setup Lambert Conformal basemap.
fig = Basemap(width=10000000,height=8000000,projection='lcc',resolution='c',lat_0=10,lon_0=90.,suppress_ticks=True)

#draw coasts
fig.drawcoastlines()

# draw boundary, fill background.
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

# draw parallels
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)

# draw meridians
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)

#translate coords into map coord system to plot

#Runway Locs
x,y = fig(runway_lons,runway_lats)

#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)

#Inmarsat Satellite Loc
x3,y3 = fig(sat_lon,sat_lat)

# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

#make legend
legend = plt.legend(loc='lower center',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#add title
plt.title('Landable Runways for a Boeing 777', fontsize=30)

#show below
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln25.svg')
plt.savefig(fn)

################################################################################################################################################################################################ 

#Inmarsat satellite height off ground

sat_height = 42170 #m
elevation_angle = np.radians(40) #elevation angle of satellite; convert degrees to radians
earth_radius = 6371 #in km, http://en.wikipedia.org/wiki/Earth

#To get the satellite calculate we have to solve several equations which were computed on Mathematica

"""
Computes the ping arc distance from the satellite to the plane.
Returns the angle in degrees.
"""
def satellite_calc(radius,orbit,angle):
    interim = (np.sqrt(-radius**2+orbit**2*(1./np.cos(angle)**2))-orbit*np.tan(angle))/np.float(orbit+radius)
    return np.degrees(2*np.arctan(interim))

ping_arc_dist = satellite_calc(earth_radius,sat_height,elevation_angle)
print "ping arc distance in degrees:", ping_arc_dist

dist_from_sat = earth_radius*np.radians(satellite_calc(earth_radius,sat_height,elevation_angle))
print "distance from satellite", dist_from_sat

################################################################################################################################################################################################ 

"""
write circle function. generates x&y pairs in a circle, 360 degrees
angle in degrees, number of points to put in circle, satellite location lat/lon
"""
def make_circle1(radius,pts,lon_loc,lat_loc): 
    coords_array = np.zeros((pts, 2))
    for i in xrange(pts):
        coords_array[i][0] =  radius*np.cos(np.radians(i)) + lat_loc 
        coords_array[i][1] =  radius*np.sin(np.radians(i)) + lon_loc 
    return coords_array

"""
write ellipse function
since across the earth it's not actually a circle
derived from spherical trigonometry
"""
def make_circle(radius,pts,lon_loc,lat_loc):
    coords_array = np.zeros((pts, 2))
    for i in xrange(pts):
        coords_array[i][0] =  radius*np.cos(np.radians(i)) + lat_loc 
        coords_array[i][1] =  radius*np.sin(np.radians(i))/(np.cos(np.cos(np.radians(radius))*(lat_loc+radius*np.cos(np.radians(i))*(np.pi/180.)))) + lon_loc 
    return coords_array

"""
write function that plots final destination from plane, from the point of the last ping.
this will be some period of time in between 0 minutes and an hour -- or else it would have pinged again.
make a point at a distance on a heading to see where the plane would be if it continued on a straight line,
from the 5th ping.
"""
def make_vector(ang_radius,heading,lon_loc,lat_loc):
    vec_lon = ang_radius*np.cos(np.radians(heading)) + lat_loc
    vec_lat = ang_radius*np.sin(np.radians(heading))/(np.cos(np.cos(np.radians(ang_radius))*(lat_loc+ang_radius*np.cos(np.radians(heading))*(np.pi/180.)))) + lon_loc
    return vec_lat,vec_lon

################################################################################################################################################################################################ 

#testing the two types of circles. Using input from:
#normal_prob_step(255.136, 30, 98.940833, 5.680556, 8.12971613367), from below,
#which initially did not work for the ellipse-like approximation.

test = make_circle(8.12971613367,360,98.940833,5.680556)

test_lat = []
test_lon = []
for i in xrange(len(test)):
    test_lat.append(test[i][0])
    test_lon.append(test[i][1])

test1 = make_circle1(8.12971613367,360,98.940833,5.680556)

test_lat1 = []
test_lon1 = []
for i in xrange(len(test1)):
    test_lat1.append(test1[i][0])
    test_lon1.append(test1[i][1])
#print test_lat

#create figure
plt.figure()
plt.plot(test_lon1,test_lat1,color='red',label='simple circle')
plt.plot(test_lon,test_lat,color='blue',label='ellipsoid')
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,markerscale=1,prop={'size':5})
plt.title('comparing circle approximations to each other')
plt.legend()
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln28.svg')
plt.savefig(fn)

################################################################################################################################################################################################ 

circle_pts = make_circle(ping_arc_dist,360,64.5,0)
#print circle_pts

circle_lat = []
for i in xrange(len(circle_pts)):
    circle_lat.append(circle_pts[i][0])

circle_lon = []
for i in xrange(len(circle_pts)):
    circle_lon.append(circle_pts[i][1])

print circle_lat[0:10]
print "-------------------"
print circle_lon[0:10]

################################################################################################################################################################################################ 

err1_20per = 0.8*ping_arc_dist
err2_20per = 1.2*ping_arc_dist

circle_pts_err1_20per = make_circle(err1_20per,360,64.5,0)
circle_pts_err2_20per = make_circle(err2_20per,360,64.5,0)

circle_lon_err1_20per = []
for i in xrange(len(circle_pts_err1_20per)):
    circle_lon_err1_20per.append(circle_pts_err1_20per[i][1])
    
circle_lon_err2_20per = []
for i in xrange(len(circle_pts_err2_20per)):
    circle_lon_err2_20per.append(circle_pts_err2_20per[i][1])
    
circle_lat_err1_20per = []
for i in xrange(len(circle_pts_err1_20per)):
    circle_lat_err1_20per.append(circle_pts_err1_20per[i][0])
    
circle_lat_err2_20per = []
for i in xrange(len(circle_pts_err2_20per)):
    circle_lat_err2_20per.append(circle_pts_err2_20per[i][0])

################################################################################################################################################################################################ 
    
print circle_lat_err1_20per[0:10]
print "--------------------"
print circle_lon_err1_20per[0:10]
print "--------------------"
print circle_lat_err2_20per[0:10]
print "--------------------"
print circle_lon_err2_20per[0:10]

################################################################################################################################################################################################ 

err1_10per = 0.9*ping_arc_dist
err2_10per = 1.1*ping_arc_dist

circle_pts_err1_10per = make_circle(err1_10per,360,64.5,0)
circle_pts_err2_10per = make_circle(err2_10per,360,64.5,0)

circle_lon_err1_10per = []
for i in xrange(len(circle_pts_err1_10per)):
    circle_lon_err1_10per.append(circle_pts_err1_10per[i][1])
    
circle_lon_err2_10per = []
for i in xrange(len(circle_pts_err2_10per)):
    circle_lon_err2_10per.append(circle_pts_err2_10per[i][1])
    
circle_lat_err1_10per = []
for i in xrange(len(circle_pts_err1_10per)):
    circle_lat_err1_10per.append(circle_pts_err1_10per[i][0])
    
circle_lat_err2_10per = []
for i in xrange(len(circle_pts_err2_10per)):
    circle_lat_err2_10per.append(circle_pts_err2_10per[i][0])
    
################################################################################################################################################################################################ 
    
err1_5per = 0.95*ping_arc_dist
err2_5per = 1.05*ping_arc_dist

circle_pts_err1_5per = make_circle(err1_5per,360,64.5,0)
circle_pts_err2_5per = make_circle(err2_5per,360,64.5,0)

circle_lon_err1_5per = []
for i in xrange(len(circle_pts_err1_5per)):
    circle_lon_err1_5per.append(circle_pts_err1_5per[i][1])
    
circle_lon_err2_5per = []
for i in xrange(len(circle_pts_err2_5per)):
    circle_lon_err2_5per.append(circle_pts_err2_5per[i][1])
    
circle_lat_err1_5per = []
for i in xrange(len(circle_pts_err1_5per)):
    circle_lat_err1_5per.append(circle_pts_err1_5per[i][0])
    
circle_lat_err2_5per = []
for i in xrange(len(circle_pts_err2_5per)):
    circle_lat_err2_5per.append(circle_pts_err2_5per[i][0])
    
################################################################################################################################################################################################ 
    
#Plot the same map again ... just add the location coordinates of where it could be located

# set figure size
fig = plt.figure(figsize=[21,15])

# setup Lambert Conformal basemap.
fig = Basemap(width=10000000,height=8000000,projection='lcc',resolution='c',lat_0=5,lon_0=90.,suppress_ticks=True)

#draw coasts
fig.drawcoastlines()

# draw boundary, fill background.
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

# draw parallels
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)

# draw meridians
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)

#translate coords into map coord system to plot
#Runway Locs
x,y = fig(runway_lons,runway_lats)
#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)
#Inmarsat Satellite Loc
x3,y3 = fig(sat_lon,sat_lat)

#Add location coords
x4,y4 = fig(99.8,6.35) #show Lankawi for electrical fire scenario

#Add circle coords -- 20% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_20per,circle_lat_err1_20per)
x7,y7 = fig(circle_lon_err2_20per,circle_lat_err2_20per)
#                     10% error
x8,y8 = fig(circle_lon_err1_10per,circle_lat_err1_10per)
x9,y9 = fig(circle_lon_err2_10per,circle_lat_err2_10per)
#                      5% error
x10,y10 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x11,y11 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')
fig.plot(x4,y4,'go',markersize=10,label='Lankawi Island')

#draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5,10,20% error')
fig.plot(x7,y7,'r--',markersize=5)
fig.plot(x8,y8,'r--',markersize=5)
fig.plot(x9,y9,'r--',markersize=5)
fig.plot(x10,y10,'r--',markersize=5)
fig.plot(x11,y11,'r--',markersize=5)

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

#make legend
legend = plt.legend(loc='lower center',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#add title
plt.title('Inmarsat Ping Estimation', fontsize=30)

#show below
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln34.svg')
plt.savefig(fn)

################################################################################################################################################################################################ 

"""
ping_prob is specifically the 5th ping probability, which we have.
we center a normal probability distribution upon the location of the radius line.

d = dist_from_sat in my case and is more abstractly 'd', a distance
r = earth_radius in my case and is is more abstractly 'r', a radius
lat1 = sat latitude in my case
lon1 = sat longitude in my case
lat2, lon2 we iterate through for our function
err is the 5,10,or 20% error Inmarsat error
"""
def ping_prob(lat1,lon1,lat2,lon2,err,d,r): 
    return np.exp(-0.5*((haversine(r,lat1,lon1,lat2,lon2)-d)/(err*d))**2)/(d*np.sqrt(2*np.pi)) 
    #model a normal distribution above...manually...could also probably use:
    #stats.norm.pdf or something of that sort
    
################################################################################################################################################################################################     

#create lat/long grid of distance
lat_max = 55 #55 N
lat_min = -50 #50 S

lon_max = 130 #130 E
lon_min = 50 #50 E

lat_range = np.abs(lat_max)+np.abs(lat_min)
lon_range = np.abs(lon_max)-np.abs(lon_min)
print "latitude range in degrees:", lat_range
print "longitude range in degrees:", lon_range

################################################################################################################################################################################################     

# x coord = longitude
# y coord = latitude
mult_fac = 1 #Use to make a finer grid, so 10 is 1050 x 800, and fill with probabilities. CHANGE TO 10 EVENTUALLY

prob_grid = np.zeros((lat_range*mult_fac,lon_range*mult_fac)) #initialize grid as numpy array

for i in xrange(lat_range*mult_fac): #across all lat grid-lets
    for j in xrange(lon_range*mult_fac): #across all lon grid-lets (so now we're iterating through rows + columns)
        prob_grid[i][j] = ping_prob(sat_lat,sat_lon,(lat_min+i/np.float(mult_fac)),(lon_min+j/np.float(mult_fac)),0.2,dist_from_sat,earth_radius) 
        #assuming 20% error == 0.2
   
################################################################################################################################################################################################          
        
#print prob_grid.shape
#print prob_grid
#print np.sum(prob_grid)

################################################################################################################################################################################################          

plt.figure()
plt.plot(prob_grid)
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln39.svg')
plt.savefig(fn)

################################################################################################################################################################################################          

#keep in mind these are *indices* not actual lat/lons
plt.figure()
plt.contour(prob_grid)
plt.title('contour plot of probabilities')
plt.ylim([0,100])
plt.xlim([0,75])
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln40.svg')
plt.savefig(fn)

################################################################################################################################################################################################          

#test out with just longitude at 0 latitude
prob_grid_lon =  np.zeros((lon_range*mult_fac))

for j in xrange(lon_range*mult_fac): #across all lon grid-lets (so now we're iterating through rows + columns)
    prob_grid_lon[j] = ping_prob(sat_lat,sat_lon,0,(lon_min+j/np.float(mult_fac)),0.2,dist_from_sat,earth_radius) #assuming 20% error == 0.2

plt.figure()
plt.plot(prob_grid_lon)
plt.title('longitude ping probability across the equator (0 deg lat)')
plt.xlabel('array index -- to get longitude need to add 50, which is minimum longitude')
plt.ylabel('probability')
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln41.svg')
plt.savefig(fn)

################################################################################################################################################################################################          

eq_deg_km = 111.32 # number of km/degree at eq Source: http://en.wikipedia.org/wiki/Decimal_degrees

################################################################################################################################################################################################          

"""
takes in a heading, a starting location, and a std dev for picking a new heading
a std dev of 30, for example, with a normal distribution, means that -30 to 30 degrees will be
where most of the new heading draws will come out of (a std dev applies to both sides of the distribution.) 
and the highest probability will be straight ahead. this is of course not including the ping probability.
"""
def normal_prob_step(old_heading, std_dev, start_lon, start_lat, ang_dist):
    
    angle_list = range(0,360) # ; print angle_list
    
    #we create a radius of 360 points that would be the heading for 360 possible degrees
    circle = make_circle(ang_dist,len(angle_list),start_lon,start_lat)
    
    #Test Suite -- uncomment to show the circle that's generated
    #plt.figure()
    #plt.plot(circle)
    #plt.title('circle')
    #plt.show()
    
    weights = np.zeros(len(angle_list)) #make 360 array
    for i in xrange(len(weights)):
        weights[i] = np.exp(-(((np.float(angle_list[i])-180.)/std_dev)**2) /2.) / (std_dev*np.sqrt(2*np.pi)) 
        #makes array of normally distributed weights as if heading was 180 degrees. Sort of a hack to make it periodic.  
        
    #Test Suite -- uncomment to show the weightings   
    #plt.figure()
    #plt.plot(weights) #check weights
    #plt.title('weights')
    #plt.show()
    
    #Now we have to translate it back to whatever the heading should be, instead of 180 degrees
    #Fortunately we can use numpy's roll. Implementing our own would be a pain.
    s_weights = np.roll(weights,-180+np.int(old_heading))
    
    #Test Suite -- uncomment to show the weightings taking into account the headings too.   
    #plt.figure()
    #plt.plot(s_weights) #check s_weights
    #plt.title('s_weights')
    #plt.show()
    
    #initialize new possible coordinates within an hr's distance, new weights for the odds, and new angles
    #(depending on whether the plane would go off the grid or not)
    new_circle = []
    new_weights = []
    new_angles = []
    
    #make sure lat & lon are in bounds
    for i in xrange(len(circle)):
        if circle[i][0] >= -50 and circle[i][0] <= 50 and circle[i][1] >= 50 and circle[i][1] <= 130:
            new_circle.append(circle[i])       
            new_weights.append(s_weights[i])
            new_angles.append(angle_list[i])
    
    #Test Suite -- uncomment to show the new circle that's generated
    #plt.figure()
    #plt.plot(new_circle)
    #plt.title('new_circle')
    #plt.show()
    
    return new_circle,new_weights,new_angles

################################################################################################################################################################################################          

"""
now need a function which given a list of discrete probabilities for each destination point, 
it will choose one of those points.

heading_init -- initial direction was headed at last known point
lon_init,lat_init -- last known point of plane in longitude and latitude
km_hop -- how far the plane went in the time interval, 1 hr. So in simplest case, the 777's cruising speed/hr.
std_dev -- the standard deviation of the heading, based on a normal distribution from the current heading (0 deg).
ping_percent_err -- what % error you assume in the Inmarsat 5th ping

"""
def five_hop_model_final(heading_init,lon_init,lat_init,km_hop,std_dev,ping_percent_err):   
    
    #initialize
    plane_lat = np.zeros(5) #initialize plane location after each hop (assumed to be 1 hr for now)
    plane_lon = np.zeros(5)  
    lat = lat_init
    lon = lon_init
    heading = heading_init
        
    for i in xrange(len(plane_lat)):
        new_circle,new_weights,new_angles = normal_prob_step(heading,std_dev,lon,lat,km_hop/eq_deg_km)
        #new_circle gives up possible coords for diff headings
        
        #Test Suite -- make sure that new_circle generates properly
        #plt.figure()
        #plt.plot(new_circle)
        #plt.title('new_circle')
        #plt.show()
        
        pp = []
        raw_weights = np.zeros(len(new_circle))
        for j in xrange(len(new_circle)):
            pp.append(ping_prob(sat_lat,sat_lon,new_circle[j][0],new_circle[j][1],ping_percent_err,dist_from_sat,earth_radius))
            raw_weights[j] = new_weights[j]*ping_prob(sat_lat,sat_lon,new_circle[j][0],new_circle[j][1],ping_percent_err,dist_from_sat,earth_radius) 
        
        probs = raw_weights / np.sum(raw_weights) #normalize
        pp2 = pp / np.sum(pp)
        
        #Test Suite -- make sure probabilites & weights change appropriately
        #plt.figure()
        #plt.plot(pp2,color='red')
        #plt.plot(new_weights,color='green')
        #plt.plot(probs,color='blue') #check probs
        #plt.show()        
        
        index = range(len(new_circle))
        chosen = np.random.choice(index,size=None,p=probs)
        #print "chosen",chosen
        
        heading = new_angles[chosen] #update heading
        #print heading
        
        plane_lat[i],plane_lon[i] = new_circle[chosen] #update position
        lat = plane_lat[i]
        lon = plane_lon[i]
    
    #at end of simulation, run the last location & heading for plane for 4 different times
    route1 = make_vector(0.25*km_hop/eq_deg_km,heading,lon,lat)
    route2 = make_vector(0.5*km_hop/eq_deg_km,heading,lon,lat)
    route3 = make_vector(0.75*km_hop/eq_deg_km,heading,lon,lat)
    route4 = make_vector((59./60.)*km_hop/eq_deg_km,heading,lon,lat)

    new_plane_lat = np.zeros(10)
    new_plane_lon = np.zeros(10)
    
    for i in xrange(len(plane_lat)):
        new_plane_lat[i] = plane_lat[i]
        new_plane_lon[i] = plane_lon[i]
    
    new_plane_lat[5] = route1[1]
    new_plane_lat[6] = route2[1]
    new_plane_lat[7] = route3[1]
    new_plane_lat[8] = route4[1]
    new_plane_lon[5] = route1[0]
    new_plane_lon[6] = route2[0]
    new_plane_lon[7] = route3[0]
    new_plane_lon[8] = route4[0]
    
    return new_plane_lat,new_plane_lon

################################################################################################################################################################################################          

last_known_heading = 255.136
km_hop = 905 #assuming 1 hr intervals, at 905 km/hr which is 777 cruising speed -- use for test case
             # max speed of a Boeing 777 is 950 km/hr FYI

N = 1000
plane_hops = []

for i in xrange(N):
    plane_hops.append(five_hop_model_final(last_known_heading,pulauperak_coord[1],pulauperak_coord[0],km_hop,30,0.2))
    
################################################################################################################################################################################################          
    
first_lat = []
two_lat = []
three_lat = []
four_lat = []
final_lat = []

route1_lat = []
route2_lat = []
route3_lat = []
route4_lat = []

first_lon = []
two_lon = []
three_lon = []
four_lon = []
final_lon = []

route1_lon = []
route2_lon = []
route3_lon = []
route4_lon = []

for i in xrange(len(plane_hops)):
    first_lat.append(plane_hops[i][0][0])
    first_lon.append(plane_hops[i][1][0])
    two_lat.append(plane_hops[i][0][1])
    two_lon.append(plane_hops[i][1][1])
    three_lat.append(plane_hops[i][0][2])
    three_lon.append(plane_hops[i][1][2])
    four_lat.append(plane_hops[i][0][3])
    four_lon.append(plane_hops[i][1][3])
    final_lat.append(plane_hops[i][0][4])
    final_lon.append(plane_hops[i][1][4])
    
    route1_lat.append(plane_hops[i][0][5])
    route1_lon.append(plane_hops[i][1][5])
    route2_lat.append(plane_hops[i][0][6])
    route2_lon.append(plane_hops[i][1][6])
    route3_lat.append(plane_hops[i][0][7])
    route3_lon.append(plane_hops[i][1][7])
    route4_lat.append(plane_hops[i][0][8])
    route4_lon.append(plane_hops[i][1][8])
    
#final dest
sim_lats_final = []
sim_lons_final = []

for i in xrange(len(plane_hops)):
    sim_lats_final.append(plane_hops[i][0][4])
    sim_lons_final.append(plane_hops[i][1][4])
    
################################################################################################################################################################################################              

#Test Suite -- uncomment to print to make sure values look right

#print final_lat
#print final_lon
#print "-------------"

#print route1_lat
#print "-------------"
#print route1_lon

#print "-------------"
#print len(plane_hops)
#print "lat", plane_hops[0][0]
#print "lon", plane_hops[0][1]
#print "lat", plane_hops[0][0][4]
#print "lon", plane_hops[0][1][4]
#print "-------------"

################################################################################################################################################################################################              

#POSITION OVER TIME

# set figure size
fig = plt.figure(figsize=[21,15])

# setup Lambert Conformal basemap.
fig = Basemap(width=10000000,height=9000000,projection='lcc',resolution='c',lat_0=-10,lon_0=90.,suppress_ticks=True)

#draw coasts
fig.drawcoastlines()

# draw boundary, fill background.
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

# draw parallels
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)

# draw meridians
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)

#translate coords into map coord system to plot
#Runway Locs
x,y = fig(runway_lons,runway_lats)
#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)
#Inmarsat Satellite Loc
x3,y3 = fig(sat_lon,sat_lat)

#Add circle coords
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_20per,circle_lat_err1_20per)
x7,y7 = fig(circle_lon_err2_20per,circle_lat_err2_20per)

#Add monte carlo sim coords
x8,y8 = fig(sim_lons_final,sim_lats_final)

#add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#add ultimate locations of MH370
x14,y14 = fig(route1_lon,route1_lat)
x15,y15 = fig(route2_lon,route2_lat)
x16,y16 = fig(route3_lon,route3_lat)
x17,y17 = fig(route4_lon,route4_lat)

# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')
#draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5)
fig.plot(x7,y7,'r--',markersize=5,label='with 20% error')
#add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='1/4 hr')
fig.plot(x15,y15,'bo',markersize=5,label='1/2 hr')
fig.plot(x16,y16,'bo',markersize=5,label='3/4 hr')
fig.plot(x17,y17,'bo',markersize=5,label='59/60 hr')

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

#make legend
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,title='Legend',markerscale=1,prop={'size':10})
legend.get_title().set_fontsize('20')

#add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#show below
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln48.svg')
plt.savefig(fn)

################################################################################################################################################################################################              

# W/OUT FINAL PLACE BUT WITH CIRCLE

# set figure size
fig = plt.figure(figsize=[21,15])

# setup Lambert Conformal basemap.
fig = Basemap(width=10000000,height=9000000,projection='lcc',resolution='c',lat_0=-10,lon_0=90.,suppress_ticks=True)

#draw coasts
fig.drawcoastlines()

# draw boundary, fill background.
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

# draw parallels
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)

# draw meridians
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)

#translate coords into map coord system to plot
#Runway Locs
x,y = fig(runway_lons,runway_lats)
#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)
#Inmarsat Satellite Loc
x3,y3 = fig(sat_lon,sat_lat)

#Add circle coords
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_20per,circle_lat_err1_20per)
x7,y7 = fig(circle_lon_err2_20per,circle_lat_err2_20per)

#Add monte carlo sim coords
x8,y8 = fig(sim_lons_final,sim_lats_final)

#add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#PLOT TEST CIRCLE
x14,y14 = fig(test_lon,test_lat)

# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')
#draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5)
fig.plot(x7,y7,'r--',markersize=5,label='with 20% error')
#add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#PLOT TEST CIRCLE
fig.plot(x14,y14,'b--',markersize=5,label='possible locations after 1 hr depending on chosen heading')

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

#make legend
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,title='Legend',markerscale=1,prop={'size':10})
legend.get_title().set_fontsize('20')

#add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#show below
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln49.svg')
plt.savefig(fn)

################################################################################################################################################################################################              

last_known_heading = 255.136
km_hop = 905 #assuming 1 hr intervals, at 905 km/hr which is 777 cruising speed -- use for test case
             # max speed is 950 km/hr FYI

N = 1000
plane_hops = []

for i in xrange(N):
    plane_hops.append(five_hop_model_final(last_known_heading,pulauperak_coord[1],pulauperak_coord[0],km_hop,30,0.1))
    
################################################################################################################################################################################################                
    
first_lat = []
two_lat = []
three_lat = []
four_lat = []
final_lat = []

route1_lat = []
route2_lat = []
route3_lat = []
route4_lat = []

first_lon = []
two_lon = []
three_lon = []
four_lon = []
final_lon = []

route1_lon = []
route2_lon = []
route3_lon = []
route4_lon = []

for i in xrange(len(plane_hops)):
    first_lat.append(plane_hops[i][0][0])
    first_lon.append(plane_hops[i][1][0])
    two_lat.append(plane_hops[i][0][1])
    two_lon.append(plane_hops[i][1][1])
    three_lat.append(plane_hops[i][0][2])
    three_lon.append(plane_hops[i][1][2])
    four_lat.append(plane_hops[i][0][3])
    four_lon.append(plane_hops[i][1][3])
    final_lat.append(plane_hops[i][0][4])
    final_lon.append(plane_hops[i][1][4])
    
    route1_lat.append(plane_hops[i][0][5])
    route1_lon.append(plane_hops[i][1][5])
    route2_lat.append(plane_hops[i][0][6])
    route2_lon.append(plane_hops[i][1][6])
    route3_lat.append(plane_hops[i][0][7])
    route3_lon.append(plane_hops[i][1][7])
    route4_lat.append(plane_hops[i][0][8])
    route4_lon.append(plane_hops[i][1][8])
    
#final dest
sim_lats_final = []
sim_lons_final = []

for i in xrange(len(plane_hops)):
    sim_lats_final.append(plane_hops[i][0][4])
    sim_lons_final.append(plane_hops[i][1][4])
    
################################################################################################################################################################################################                

#POSITION OVER TIME

# set figure size
fig = plt.figure(figsize=[21,15])

# setup Lambert Conformal basemap.
fig = Basemap(width=10000000,height=9000000,projection='lcc',resolution='c',lat_0=-10,lon_0=90.,suppress_ticks=True)

#draw coasts
fig.drawcoastlines()

# draw boundary, fill background.
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

# draw parallels
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)

# draw meridians
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)

#translate coords into map coord system to plot
#Runway Locs
x,y = fig(runway_lons,runway_lats)
#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)
#Inmarsat Satellite Loc
x3,y3 = fig(sat_lon,sat_lat)

#Add circle coords
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_10per,circle_lat_err1_10per)
x7,y7 = fig(circle_lon_err2_10per,circle_lat_err2_10per)

#Add monte carlo sim coords
x8,y8 = fig(sim_lons_final,sim_lats_final)

#add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#add ultimate locations of MH370
x14,y14 = fig(route1_lon,route1_lat)
x15,y15 = fig(route2_lon,route2_lat)
x16,y16 = fig(route3_lon,route3_lat)
x17,y17 = fig(route4_lon,route4_lat)

# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')
#draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5)
fig.plot(x7,y7,'r--',markersize=5,label='with 10% error')
#add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='1/4 hr')
fig.plot(x15,y15,'bo',markersize=5,label='1/2 hr')
fig.plot(x16,y16,'bo',markersize=5,label='3/4 hr')
fig.plot(x17,y17,'bo',markersize=5,label='59/60 hr')

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

#make legend
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,title='Legend',markerscale=1,prop={'size':10})
legend.get_title().set_fontsize('20')

#add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#show below
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln52.svg')
plt.savefig(fn)

################################################################################################################################################################################################                

# W/OUT FINAL PLACE BUT WITH CIRCLE

# set figure size
fig = plt.figure(figsize=[21,15])

# setup Lambert Conformal basemap.
fig = Basemap(width=10000000,height=9000000,projection='lcc',resolution='c',lat_0=-10,lon_0=90.,suppress_ticks=True)

#draw coasts
fig.drawcoastlines()

# draw boundary, fill background.
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

# draw parallels
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)

# draw meridians
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)

#translate coords into map coord system to plot
#Runway Locs
x,y = fig(runway_lons,runway_lats)
#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)
#Inmarsat Satellite Loc
x3,y3 = fig(sat_lon,sat_lat)

#Add circle coords
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_10per,circle_lat_err1_10per)
x7,y7 = fig(circle_lon_err2_10per,circle_lat_err2_10per)

#Add monte carlo sim coords
x8,y8 = fig(sim_lons_final,sim_lats_final)

#add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#PLOT TEST CIRCLE
x14,y14 = fig(test_lon,test_lat)

# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')
#draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5)
fig.plot(x7,y7,'r--',markersize=5,label='with 10% error')
#add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#PLOT TEST CIRCLE
fig.plot(x14,y14,'b--',markersize=5,label='possible locations after 1 hr depending on chosen heading')

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

#make legend
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,title='Legend',markerscale=1,prop={'size':10})
legend.get_title().set_fontsize('20')

#add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#show below
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln53.svg')
plt.savefig(fn)

################################################################################################################################################################################################                

last_known_heading = 255.136
km_hop = 905 #assuming 1 hr intervals, at 905 km/hr which is 777 cruising speed -- use for test case
             # max speed is 950 km/hr FYI

N = 1000
plane_hops = []

for i in xrange(N):
    plane_hops.append(five_hop_model_final(last_known_heading,pulauperak_coord[1],pulauperak_coord[0],km_hop,30,0.05))
    
################################################################################################################################################################################################                

first_lat = []
two_lat = []
three_lat = []
four_lat = []
final_lat = []

route1_lat = []
route2_lat = []
route3_lat = []
route4_lat = []

first_lon = []
two_lon = []
three_lon = []
four_lon = []
final_lon = []

route1_lon = []
route2_lon = []
route3_lon = []
route4_lon = []

for i in xrange(len(plane_hops)):
    first_lat.append(plane_hops[i][0][0])
    first_lon.append(plane_hops[i][1][0])
    two_lat.append(plane_hops[i][0][1])
    two_lon.append(plane_hops[i][1][1])
    three_lat.append(plane_hops[i][0][2])
    three_lon.append(plane_hops[i][1][2])
    four_lat.append(plane_hops[i][0][3])
    four_lon.append(plane_hops[i][1][3])
    final_lat.append(plane_hops[i][0][4])
    final_lon.append(plane_hops[i][1][4])
    
    route1_lat.append(plane_hops[i][0][5])
    route1_lon.append(plane_hops[i][1][5])
    route2_lat.append(plane_hops[i][0][6])
    route2_lon.append(plane_hops[i][1][6])
    route3_lat.append(plane_hops[i][0][7])
    route3_lon.append(plane_hops[i][1][7])
    route4_lat.append(plane_hops[i][0][8])
    route4_lon.append(plane_hops[i][1][8])
    
#final dest
sim_lats_final = []
sim_lons_final = []

for i in xrange(len(plane_hops)):
    sim_lats_final.append(plane_hops[i][0][4])
    sim_lons_final.append(plane_hops[i][1][4])
    
################################################################################################################################################################################################                
    
#POSITION OVER TIME

# set figure size
fig = plt.figure(figsize=[21,15])

# setup Lambert Conformal basemap.
fig = Basemap(width=10000000,height=9000000,projection='lcc',resolution='c',lat_0=-10,lon_0=90.,suppress_ticks=True)

#draw coasts
fig.drawcoastlines()

# draw boundary, fill background.
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

# draw parallels
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)

# draw meridians
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)

#translate coords into map coord system to plot
#Runway Locs
x,y = fig(runway_lons,runway_lats)
#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)
#Inmarsat Satellite Loc
x3,y3 = fig(sat_lon,sat_lat)

#Add circle coords
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add monte carlo sim coords
x8,y8 = fig(sim_lons_final,sim_lats_final)

#add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#add ultimate locations of MH370
x14,y14 = fig(route1_lon,route1_lat)
x15,y15 = fig(route2_lon,route2_lat)
x16,y16 = fig(route3_lon,route3_lat)
x17,y17 = fig(route4_lon,route4_lat)

# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')
#draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5)
fig.plot(x7,y7,'r--',markersize=5,label='with 5% error')
#add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='1/4 hr')
fig.plot(x15,y15,'bo',markersize=5,label='1/2 hr')
fig.plot(x16,y16,'bo',markersize=5,label='3/4 hr')
fig.plot(x17,y17,'bo',markersize=5,label='59/60 hr')

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

#make legend
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,title='Legend',markerscale=1,prop={'size':10})
legend.get_title().set_fontsize('20')

#add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#show below
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln56.svg')
plt.savefig(fn)

################################################################################################################################################################################################                

# W/OUT FINAL PLACE BUT WITH CIRCLE

# set figure size
fig = plt.figure(figsize=[21,15])

# setup Lambert Conformal basemap.
fig = Basemap(width=10000000,height=9000000,projection='lcc',resolution='c',lat_0=-10,lon_0=90.,suppress_ticks=True)

#draw coasts
fig.drawcoastlines()

# draw boundary, fill background.
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

# draw parallels
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)

# draw meridians
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)

#translate coords into map coord system to plot
#Runway Locs
x,y = fig(runway_lons,runway_lats)
#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)
#Inmarsat Satellite Loc
x3,y3 = fig(sat_lon,sat_lat)

#Add circle coords
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add monte carlo sim coords
x8,y8 = fig(sim_lons_final,sim_lats_final)

#add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#PLOT TEST CIRCLE
x14,y14 = fig(test_lon,test_lat)

# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')
#draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5)
fig.plot(x7,y7,'r--',markersize=5,label='with 5% error')
#add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#PLOT TEST CIRCLE
fig.plot(x14,y14,'b--',markersize=5,label='possible locations after 1 hr depending on chosen heading')

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

#make legend
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,title='Legend',markerscale=1,prop={'size':10})
legend.get_title().set_fontsize('20')

#add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#show below
#plt.show()

fn = os.path.join(os.path.dirname(__file__), 'plot_ln57.svg')
plt.savefig(fn)