description=u'''
Created on Apr 1, 2014

Nat Laughlin
http://natlaughlin.com
http://github.com/natlaughlin

Based on:

http://qa.nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-V2-Part1.ipynb?create=1
http://qa.nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-V2-Part2.ipynb?create=1
http://qa.nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-V2-Part3.ipynb?create=1

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

################################################################################################################################################################################################                

#read cli params

import os
import optparse

class PlainHelpFormatter(optparse.IndentedHelpFormatter): 
    def format_description(self, description):
        if description:
            return description + "\n"
        else:
            return ""

default = 255.136
parser = optparse.OptionParser(formatter=PlainHelpFormatter(),description=description)



help = """last_known_heading (default: {0})
calculated in Mathematica from MH370's two last publicly known locations:
when it deviated from its flight path, and when it was last detected by Malaysian military radar
0 degrees is due north, so this is basically to the west (270 degrees), but slightly south
""".format(default)
parser.add_option("-l", "--last-known-heading", dest="last_known_heading", default=default, help=help)

default = 905
help = """km_hop (default: {0})
assuming 1 hr intervals, at 905 km/hr which is 777 cruising speed -- use for test case
max speed of a Boeing 777 is 950 km/hr FYI
""".format(default)
parser.add_option("-k", "--km-hop", dest="km_hop", default=default, help=help)

default = 1000
help = """N (default: {0})
define number of simulations to run
""".format(default)
parser.add_option("-n", "--simulations", dest="simulations", default=default, help=help)

default = 30
help = """std_dev (default: {0})
the standard deviation is the only arbitrary choice for us, along with the N=1000 simulations.
but it fits the notion that the plane is likely to continue on the same course, allowing for
some turns / heading change over the course of an hour. I stand by this number, although it's easy
to change to 15 or 45 and see what the difference is. The smaller the number the narrower the
ability to change heading is. The larger the number the more like a true "random walk" the plane's
direction and by consequence, location will be, at each time step.
""".format(default)
parser.add_option("-s", "--std-dev", dest="std_dev", default=default, help=help)

default = os.path.dirname(__file__)
help = """output directory (default: {0})
""".format(default)
parser.add_option('-o', '--output-directory', dest='output_directory', default=default, help=help)

(options, args) = parser.parse_args()

if not os.path.exists(options.output_directory):
    os.makedirs(options.output_directory)
    
parsed_options = ", ".join(['{0}: {1}'.format(key, value) for (key, value) in options.__dict__.items()])

def savefig(plt, file):
    fn = os.path.join(options.output_directory, file)
    plt.figtext(0,1,parsed_options, fontsize=10, horizontalalignment='left', verticalalignment='top')
    plt.savefig(fn)
    plt.close()

################################################################################################################################################################################################                
# PART 1
################################################################################################################################################################################################                
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
import matplotlib.pyplot as plt
import seaborn as sns
#from IPython import display
from mpl_toolkits.basemap import Basemap # use for plotting lat lon & world maps


################################################################################################################################################################################################                

eq_deg_km = 111.32 # number of km/degree at eq Source: http://en.wikipedia.org/wiki/Decimal_degrees
earth_radius = 6371 #in km, http://en.wikipedia.org/wiki/Earth

#Inmarsat satellite information
sat_height = 42170 #Inmarsat satellite height off ground, in meters
elevation_angle = np.radians(40) #elevation angle of satellite; convert degrees to radians. Source: NYT Hong Kong Bureau

################################################################################################################################################################################################                

# The Inmarsat satellite is at 0,64.5 -- it's geostationary.
inmarsat = [0, 64.5]

#Now we plot the plane's known positions

#Kuala Lumpur International Airport Coordinates: http://www.distancesfrom.com/my/Kuala-Lumpur-Airport-(KUL)-Malaysia-latitude-longitude-Kuala-Lumpur-Airport-(KUL)-Malaysia-latitude-/LatLongHistory/3308940.aspx
kualalumpur = [2.7544829, 101.7011363]

#Pulau Perak coordinates: http://tools.wmflabs.org/geohack/geohack.php?pagename=Pulau_Perak&params=5_40_50_N_98_56_27_E_type:isle_region:MY
# http://en.wikipedia.org/wiki/Perak_Island -> Indonesia military radar detected near island
pulauperak = [5.680556, 98.940833]

# Igari Waypoint. Source: # http://www.fallingrain.com/waypoint/SN/IGARI.html Given in hours,minutes,seconds.
igariwaypoint = [6. + 56./60. + 12./3600., 103. + 35./60. + 6./3600.] 

print "inmarsat lat/lon:", inmarsat[0], inmarsat[1]
print "kuala lumpur int'l airport coord lat/lon:", kualalumpur[0],kualalumpur[1]
print "pulau perak lat/lon:", pulauperak[0],pulauperak[1]
print "igari waypoint lat/lon:", igariwaypoint[0],igariwaypoint[1]

################################################################################################################################################################################################                

#create lat/long grid of distance

lat_min = -50 #50 S
lat_max = 50  #50 N
lon_min = 50  #50 E
lon_max = 140 #130 E
lat_space = 5 #spacing for plotting latitudes and longitudes
lon_space = 5

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=13000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='lower left',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Initial Map & Data', fontsize=30)

#Show below
#plt.show()

file = 'part1_plot_ln5.svg'
savefig(plt, file)

################################################################################################################################################################################################                

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
        coords_array[i][1] =  radius*np.sin(np.radians(i))/(np.cos(np.cos(np.radians(radius))*(lat_loc+radius*np.cos(np.radians(i)))*(np.pi/180.))) + lon_loc 
    return coords_array

################################################################################################################################################################################################                

#test near the equator, -5 (5 S)

test = make_circle(8.12971613367,360,90,-5)

test_lat = []
test_lon = []

for i in xrange(len(test)):
    test_lat.append(test[i][0])
    test_lon.append(test[i][1])

test1 = make_circle1(8.12971613367,360,90,-5)

test_lat1 = []
test_lon1 = []

for i in xrange(len(test1)):
    test_lat1.append(test1[i][0])
    test_lon1.append(test1[i][1])

#create figure
plt.figure()
plt.plot(test_lon1,test_lat1,color='red',label='simple circle')
plt.plot(test_lon,test_lat,color='blue',label='ellipsoid')
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,markerscale=1,prop={'size':5})
plt.title('comparing circle approximations to each other')
plt.legend()
#plt.show()

file = 'part1_plot_ln8.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#test at a low latitude, -40 (40 S)

test = make_circle(8.12971613367,360,90,-40)

test_lat = []
test_lon = []

for i in xrange(len(test)):
    test_lat.append(test[i][0])
    test_lon.append(test[i][1])

test1 = make_circle1(8.12971613367,360,90,-40)

test_lat1 = []
test_lon1 = []

for i in xrange(len(test1)):
    test_lat1.append(test1[i][0])
    test_lon1.append(test1[i][1])

#create figure
plt.figure()
plt.plot(test_lon1,test_lat1,color='red',label='simple circle')
plt.plot(test_lon,test_lat,color='blue',label='ellipsoid')
legend = plt.legend(loc='lower left',fontsize=8,frameon=True,markerscale=1,prop={'size':5})
plt.title('comparing circle approximations to each other')
plt.legend()
#plt.show()

file = 'part1_plot_ln9.svg'
savefig(plt, file)

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

print "Percent error 1 way is:", (1./40)*100
print "Percent error round trip is:", (2./40)*100

################################################################################################################################################################################################                

err1_5per = 0.95*ping_arc_dist
err2_5per = 1.05*ping_arc_dist

circle_pts_err1_5per = make_circle(err1_5per,360,64.5,0)
circle_pts_err2_5per = make_circle(err2_5per,360,64.5,0)

circle_lon_err1_5per = []
circle_lon_err2_5per = []
circle_lat_err1_5per = []
circle_lat_err2_5per = []

for i in xrange(len(circle_pts_err1_5per)):
    circle_lon_err1_5per.append(circle_pts_err1_5per[i][1])
    
for i in xrange(len(circle_pts_err2_5per)):
    circle_lon_err2_5per.append(circle_pts_err2_5per[i][1])
    
for i in xrange(len(circle_pts_err1_5per)):
    circle_lat_err1_5per.append(circle_pts_err1_5per[i][0])
    
for i in xrange(len(circle_pts_err2_5per)):
    circle_lat_err2_5per.append(circle_pts_err2_5per[i][0])

################################################################################################################################################################################################                

err1_2per = 0.975*ping_arc_dist
err2_2per = 1.025*ping_arc_dist

circle_pts_err1_2per = make_circle(err1_2per,360,64.5,0)
circle_pts_err2_2per = make_circle(err2_2per,360,64.5,0)

circle_lon_err1_2per = []
circle_lon_err2_2per = []
circle_lat_err1_2per = []
circle_lat_err2_2per = []

for i in xrange(len(circle_pts_err1_2per)):
    circle_lon_err1_2per.append(circle_pts_err1_2per[i][1])
    
for i in xrange(len(circle_pts_err2_2per)):
    circle_lon_err2_2per.append(circle_pts_err2_2per[i][1])
    
for i in xrange(len(circle_pts_err1_2per)):
    circle_lat_err1_2per.append(circle_pts_err1_2per[i][0])
    
for i in xrange(len(circle_pts_err2_2per)):
    circle_lat_err2_2per.append(circle_pts_err2_2per[i][0])

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=13000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)
x8,y8 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x9,y9 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 2.5 & 5% error')
fig.plot(x7,y7,'r--',markersize=5)
fig.plot(x8,y8,'r--',markersize=5)
fig.plot(x9,y9,'r--',markersize=5)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='lower left',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation', fontsize=30)

#Show below
#plt.show()

file = 'part1_plot_ln14.svg'
savefig(plt, file)

################################################################################################################################################################################################                

"""
Haversine equation.
Computes the great circle distance between two pairs of longitude/latitude.
Returns the distance in m or km depending on input (I use meters.) 
"""
def haversine(r,lat1,lon1,lat2,lon2):
    d = 2.0*r*np.arcsin(np.sqrt(np.sin(np.radians(lat2-lat1)/2.0)**2+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(np.radians(lon2-lon1)/2.0)**2))
    return d   

################################################################################################################################################################################################                


"""
ping_prob_normal is specifically the 5th ping probability, which we have.
we center a normal probability distribution upon the location of the radius line.

d = dist_from_sat in my case and is more abstractly 'd', a distance
r = earth_radius in my case and is is more abstractly 'r', a radius
lat1 = sat latitude in my case
lon1 = sat longitude in my case
lat2, lon2 we iterate through for our function
err is the 5,10,or 20% error Inmarsat error
"""
def ping_prob_normal(lat1,lon1,lat2,lon2,err,d,r): 
    return np.exp(-0.5*((haversine(r,lat1,lon1,lat2,lon2)-d)/(err*d))**2)/(d*np.sqrt(2*np.pi)) 

################################################################################################################################################################################################                

lat_range = np.abs(lat_max)+np.abs(lat_min)
lon_range = np.abs(lon_max)-np.abs(lon_min)
print "latitude range in degrees:", lat_range
print "longitude range in degrees:", lon_range

################################################################################################################################################################################################                

# x coord = longitude
# y coord = latitude
mult_fac = 10 #Use to make a finer grid, so 10 per 1 degree lat/lon , and fill with probabilities.

prob_grid = np.zeros((lat_range*mult_fac,lon_range*mult_fac)) #initialize grid as numpy array

for i in xrange(lat_range*mult_fac): #across all lat grid-lets
    for j in xrange(lon_range*mult_fac): #across all lon grid-lets (so now we're iterating through rows + columns)
        prob_grid[i][j] = ping_prob_normal(inmarsat[0],inmarsat[1],(lat_min+i/np.float(mult_fac)),(lon_min+j/np.float(mult_fac)),0.05,dist_from_sat,earth_radius) 
        #assuming 5% error ... 5% == 0.05

################################################################################################################################################################################################                

plt.figure()
plt.plot(prob_grid)
#plt.show()

file = 'part1_plot_ln19.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#test out with just longitude at 0 latitude
prob_grid_lon =  np.zeros((lon_range*mult_fac))

for j in xrange(lon_range*mult_fac): #across all lon grid-lets (so now we're iterating through rows + columns)
    prob_grid_lon[j] = ping_prob_normal(inmarsat[0],inmarsat[1],0,(lon_min+j/np.float(mult_fac)),0.05,dist_from_sat,earth_radius)

plt.figure()
plt.plot(prob_grid_lon)
plt.title('longitude ping probability across the equator (0 deg lat)')
plt.xlabel('array index -- to get longitude need to add 500, which is minimum longitude')
plt.ylabel('probability')
#plt.show()

file = 'part1_plot_ln20.svg'
savefig(plt, file)

################################################################################################################################################################################################                

"""
write function that plots final destination from plane, from the point of the last ping.
this will be some period of time in between 0 minutes and an hour -- or else it would have pinged again.
make a point at a distance on a heading to see where the plane would be if it continued on a straight line,
from the 5th ping.
"""
def make_vector(ang_radius,heading,lon_loc,lat_loc):
    vec_lat = ang_radius*np.cos(np.radians(heading)) + lat_loc
    vec_lon = ang_radius*np.sin(np.radians(heading))/(np.cos(np.cos(np.radians(ang_radius))*(lat_loc+ang_radius*np.cos(np.radians(heading)))*(np.pi/180.))) + lon_loc
    return vec_lat,vec_lon

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
    
    weights = np.zeros(len(angle_list)) #make 360 array
    for i in xrange(len(weights)):
        weights[i] = np.exp(-(((np.float(angle_list[i])-180.)/std_dev)**2) /2.) / (std_dev*np.sqrt(2*np.pi)) 
        #makes array of normally distributed weights as if heading was 180 degrees. Sort of a hack to make it periodic.  
    
    #Now we have to translate it back to whatever the heading should be, instead of 180 degrees
    #Fortunately we can use numpy's roll. Implementing our own would be a pain.
    s_weights = np.roll(weights,-180+np.int(old_heading))
    
    #initialize new possible coordinates within an hr's distance, new weights for the odds, and new angles
    #(depending on whether the plane would go off the grid or not)
    new_circle = []
    new_weights = []
    new_angles = []
    
    #make sure lat & lon are in bounds
    for i in xrange(len(circle)):
        if circle[i][0] >= lat_min and circle[i][0] <= lat_max and circle[i][1] >= lon_min and circle[i][1] <= lon_max:
            new_circle.append(circle[i])       
            new_weights.append(s_weights[i])
            new_angles.append(angle_list[i])

    return new_circle,new_weights,new_angles

################################################################################################################################################################################################                

"""
a function which given a list of discrete probabilities for each destination point, 
it will choose one of those points.

heading_init -- initial direction was headed at last known point
lon_init,lat_init -- last known point of plane in longitude and latitude
km_hop -- how far the plane went in the time interval, 1 hr. So in simplest case, the 777's cruising speed/hr.
std_dev -- the standard deviation of the heading, based on a normal distribution from the current heading (0 deg).
ping_percent_err -- what % error you assume in the Inmarsat 5th ping. either 2.5 or 5%.

uses normal distribution for heading

"""
def five_hop_model_normal(heading_init,lon_init,lat_init,km_hop,std_dev,ping_percent_err):   
    
    #initialize
    plane_lat = np.zeros(5) #initialize plane location after each hop (assumed to be 1 hr for now)
    plane_lon = np.zeros(5)  
    lat = lat_init
    lon = lon_init
    heading = heading_init
        
    for i in xrange(len(plane_lat)):
        new_circle,new_weights,new_angles = normal_prob_step(heading,std_dev,lon,lat,km_hop/eq_deg_km)
        #new_circle gives up possible coords for diff headings
        
        raw_weights = np.zeros(len(new_circle))
        for j in xrange(len(new_circle)):
            raw_weights[j] = new_weights[j]*ping_prob_normal(inmarsat[0],inmarsat[1],new_circle[j][0],new_circle[j][1],ping_percent_err,dist_from_sat,earth_radius) 
        
        probs = raw_weights / np.sum(raw_weights) #normalize
        
        index = range(len(new_circle))
        chosen = np.random.choice(index,size=None,p=probs)
        #print "chosen",chosen
        
        heading = new_angles[chosen] #update heading
        
        plane_lat[i],plane_lon[i] = new_circle[chosen] #update position
        lat = plane_lat[i]
        lon = plane_lon[i]
    
    #at end of simulation, run the last location & heading for plane for 4 different times
    route1 = make_vector(0.25*km_hop/eq_deg_km,heading,lon,lat)
    route2 = make_vector(0.5*km_hop/eq_deg_km,heading,lon,lat)
    route3 = make_vector(0.75*km_hop/eq_deg_km,heading,lon,lat)
    route4 = make_vector((59./60.)*km_hop/eq_deg_km,heading,lon,lat)

    new_plane_lat = np.zeros(9)
    new_plane_lon = np.zeros(9)
    
    for i in xrange(len(plane_lat)):
        new_plane_lat[i] = plane_lat[i]
        new_plane_lon[i] = plane_lon[i]
    
    new_plane_lat[5] = route1[0]
    new_plane_lat[6] = route2[0]
    new_plane_lat[7] = route3[0]
    new_plane_lat[8] = route4[0]
    new_plane_lon[5] = route1[1]
    new_plane_lon[6] = route2[1]
    new_plane_lon[7] = route3[1]
    new_plane_lon[8] = route4[1]
    
    return new_plane_lat,new_plane_lon

################################################################################################################################################################################################                

last_known_heading = 255.136 #calculated in Mathematica from MH370's two last publically known locations:
                             #when it deviated from its flight path, and when it was last detected by Malyasian military radar
                             #0 degrees is due north, so this is basically to the west (270 degrees), but slightly south

km_hop = 905 #assuming 1 hr intervals, at 905 km/hr which is 777 cruising speed -- use for test case
             # max speed of a Boeing 777 is 950 km/hr FYI

N = 1000 #define number of simulations to run

################################################################################################################################################################################################                

#override simulation params with cli options

if options.km_hop:
    km_hop = float(options.km_hop)
    
if options.last_known_heading:
    last_known_heading = float(options.last_known_heading)

if options.simulations:
    N = int(options.simulations)

################################################################################################################################################################################################                

percenterror = 0.05

std_dev = 30 #the standard deviation is the only arbitrary choice for us, along with the N=1000 simulations.
             #but it fits the notion that the plane is likely to continue on the same course, allowing for
             #some turns / heading change over the course of an hour. I stand by this number, although it's easy
             #to change to 15 or 45 and see what the difference is. The smaller the number the narrower the
             #ability to change heading is. The larger the number the more like a true "random walk" the plane's
             #direction and by consequence, location will be, at each time step.

################################################################################################################################################################################################                

#override simulation params with cli options

if options.std_dev:
    std_dev = float(options.std_dev)

################################################################################################################################################################################################                


plane_hops = []

for i in xrange(N):
    plane_hops.append(five_hop_model_normal(last_known_heading,pulauperak[1],pulauperak[0],km_hop,std_dev,percenterror))

################################################################################################################################################################################################                

first_lat = []
two_lat = []
three_lat = []
four_lat = []
final_lat = []

first_lon = []
two_lon = []
three_lon = []
four_lon = []
final_lon = []

route1_lat = []
route2_lat = []
route3_lat = []
route4_lat = []

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

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=13000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='lower left',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part1_plot_ln43.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon,route1_lat)
x15,y15 = fig(route2_lon,route2_lat)
x16,y16 = fig(route3_lon,route3_lat)
x17,y17 = fig(route4_lon,route4_lat)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part1_plot_ln45.svg'
savefig(plt, file)

################################################################################################################################################################################################                

percenterror = 0.025

std_dev = 30

################################################################################################################################################################################################                

#override simulation params with cli options

if options.std_dev:
    std_dev = float(options.std_dev)

################################################################################################################################################################################################                

plane_hops = []

for i in xrange(N):
    plane_hops.append(five_hop_model_normal(last_known_heading,pulauperak[1],pulauperak[0],km_hop,std_dev,percenterror))

################################################################################################################################################################################################                

first_lat = []
two_lat = []
three_lat = []
four_lat = []
final_lat = []

first_lon = []
two_lon = []
three_lon = []
four_lon = []
final_lon = []

route1_lat = []
route2_lat = []
route3_lat = []
route4_lat = []

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

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=13000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)

#Add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 2.5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='lower left',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part1_plot_ln50.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)

#Add points after each hr
x9,y9 = fig(first_lon,first_lat)
x10,y10 = fig(two_lon,two_lat)
x11,y11 = fig(three_lon,three_lat)
x12,y12 = fig(four_lon,four_lat)
x13,y13 = fig(final_lon,final_lat)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon,route1_lat)
x15,y15 = fig(route2_lon,route2_lat)
x16,y16 = fig(route3_lon,route3_lat)
x17,y17 = fig(route4_lon,route4_lat)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 2.5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part1_plot_ln51.svg'
savefig(plt, file)

################################################################################################################################################################################################                
# PART 2
################################################################################################################################################################################################                

"""
takes in angle and heading in degrees (note conversion to radians in function)
k goes from 0 to infinity and is a point at infinity, and a uniform distribution at 0
"""
def von_mises(angle,heading,k):
    return np.exp(k*np.cos(np.radians(angle)-np.radians(heading)))/(2*np.pi*np.i0(k))

################################################################################################################################################################################################                

circle_test = []

for i in xrange(360):
    circle_test.append(von_mises(i,150,0))
    
plt.figure()
plt.plot(circle_test)
plt.title('k=0 -- should be uniform')
#plt.show()

file = 'part2_plot_ln49_1.svg'
savefig(plt, file)

circle_test2 = []

for i in xrange(360):
    circle_test2.append(von_mises(i,150,1))

plt.figure()
plt.plot(circle_test2)
plt.title('k=1 -- should be bell curve')
#plt.show()

file = 'part2_plot_ln49_2.svg'
savefig(plt, file)

circle_test3 = []

for i in xrange(360):
    circle_test3.append(von_mises(i,150,10))
    
plt.figure()
plt.plot(circle_test3)
plt.title('k=10 -- should be contracting/converging toward a point')
#plt.show()

file = 'part2_plot_ln49_3.svg'
savefig(plt, file)

################################################################################################################################################################################################                

"""
takes in angle and heading in degrees (note conversion to radians in function)
k goes from 0-1 and is a point at 1, and a uniform distribution at 0
"""
def wrapped_cauchy(angle,heading,k):
    return np.float((1-k**2))/(1+k**2-2*k*np.cos(np.radians(angle)-np.radians(heading)))

################################################################################################################################################################################################                

circle_test = []

for i in xrange(360):
    circle_test.append(wrapped_cauchy(i,150,0))
    
plt.figure()
plt.plot(circle_test)
plt.title('k=0 -- should be uniform')
#plt.show()

file = 'part2_plot_ln51_1.svg'
savefig(plt, file)

circle_test2 = []

for i in xrange(360):
    circle_test2.append(wrapped_cauchy(i,150,0.5))

plt.figure()
plt.plot(circle_test2)
plt.title('k=0.5 -- should be bell curve')
#plt.show()

file = 'part2_plot_ln51_2.svg'
savefig(plt, file)

circle_test3 = []

for i in xrange(360):
    circle_test3.append(wrapped_cauchy(i,150,0.99))
    
plt.figure()
plt.plot(circle_test3)
plt.title('k=0.99 -- should be contracting/converging toward a point')
#plt.show()

file = 'part2_plot_ln51_3.svg'
savefig(plt, file)

################################################################################################################################################################################################                

"""
takes in a heading, a starting location, and a k for picking a new heading based on the Von Mises distribution.
k = 0 -> uniform distribution. jump all over the place, true random walk.
k = 1,10 -> similar to before. tends to go straight ahead or near-straight ahead.
            1 is like a normal distribution with a large standard deviation from the mean.
            10 is like a normal distribution with a small standard deviation from the mean.
"""
def von_mises_prob_step(heading, k, start_lon, start_lat, ang_dist):
    
    angle_list = range(0,360)
    
    #we create a radius of 360 points that would be the heading for 360 possible degrees
    circle = make_circle(ang_dist,len(angle_list),start_lon,start_lat)
    
    weights = np.zeros(len(angle_list)) #make 360 array
    for i in xrange(len(weights)):
        weights[i] = von_mises(angle_list[i],heading,k)
        #makes array of von mises distributed weights.  
    
    #initialize new possible coordinates within an hr's distance, new weights for the odds, and new angles
    new_circle = []
    new_weights = []
    new_angles = []
    
    #make sure lat & lon are in bounds
    for i in xrange(len(circle)):
        if circle[i][0] >= lat_min and circle[i][0] <= lat_max and circle[i][1] >= lon_min and circle[i][1] <= lon_max:
            new_circle.append(circle[i])       
            new_weights.append(weights[i])
            new_angles.append(angle_list[i])
    
    return new_circle,new_weights,new_angles

################################################################################################################################################################################################                

"""
takes in a heading, a starting location, and a k for picking a new heading based on the Wrapped Cauchy distribution.
k = 0 -> uniform distribution. jump all over the place, true random walk.
k = 0.5 -> similar to before. tends to go straight ahead or near-straight ahead.
k = 0.95 -> plane makes a bee-line for straight ahead. there's just no "convincing" it otherwise. 
            almost no deviation from 0 degrees relative to previous heading.
"""
def wrapped_cauchy_prob_step(heading, k, start_lon, start_lat, ang_dist):
    
    angle_list = range(0,360)
    
    #we create a radius of 360 points that would be the heading for 360 possible degrees
    circle = make_circle(ang_dist,len(angle_list),start_lon,start_lat)
    
    weights = np.zeros(len(angle_list)) #make 360 array
    for i in xrange(len(weights)):
        weights[i] = wrapped_cauchy(angle_list[i],heading,k)

    #initialize new possible coordinates within an hr's distance, new weights for the odds, and new angles
    new_circle = []
    new_weights = []
    new_angles = []
    
    #make sure lat & lon are in bounds
    for i in xrange(len(circle)):
        if circle[i][0] >= lat_min and circle[i][0] <= lat_max and circle[i][1] >= lon_min and circle[i][1] <= lon_max:
            new_circle.append(circle[i])       
            new_weights.append(weights[i])
            new_angles.append(angle_list[i])
    
    return new_circle,new_weights,new_angles

################################################################################################################################################################################################                

"""
a function which given a list of discrete probabilities for each destination point, 
it will choose one of those points.

heading_init -- initial direction was headed at last known point
lon_init,lat_init -- last known point of plane in longitude and latitude
km_hop -- how far the plane went in the time interval, 1 hr. So in simplest case, the 777's cruising speed/hr.
k -- affects the heading distribution, based on a Von Mises distribution from the current heading (0 deg).
ping_percent_err -- what % error you assume in the Inmarsat 5th ping. either 2.5 or 5%.

uses Von Mises distribution for heading

"""
def five_hop_model_von_mises(heading_init,lon_init,lat_init,km_hop,k,ping_percent_err):   
    
    #initialize
    plane_lat = np.zeros(5) #initialize plane location after each hop (assumed to be 1 hr for now)
    plane_lon = np.zeros(5)  
    lat = lat_init
    lon = lon_init
    heading = heading_init
        
    for i in xrange(len(plane_lat)):
        new_circle,new_weights,new_angles = von_mises_prob_step(heading,k,lon,lat,km_hop/eq_deg_km)
        #new_circle gives up possible coords for diff headings
        
        raw_weights = np.zeros(len(new_circle))
        for j in xrange(len(new_circle)):
            raw_weights[j] = new_weights[j]*ping_prob_normal(inmarsat[0],inmarsat[1],new_circle[j][0],new_circle[j][1],ping_percent_err,dist_from_sat,earth_radius) 
        
        probs = raw_weights / np.sum(raw_weights) #normalize
        
        index = range(len(new_circle))
        chosen = np.random.choice(index,size=None,p=probs)
        #print "chosen",chosen
        
        heading = new_angles[chosen] #update heading
        
        plane_lat[i],plane_lon[i] = new_circle[chosen] #update position
        lat = plane_lat[i]
        lon = plane_lon[i]
    
    #at end of simulation, run the last location & heading for plane for 4 different times
    route1 = make_vector(0.25*km_hop/eq_deg_km,heading,lon,lat)
    route2 = make_vector(0.5*km_hop/eq_deg_km,heading,lon,lat)
    route3 = make_vector(0.75*km_hop/eq_deg_km,heading,lon,lat)
    route4 = make_vector((59./60.)*km_hop/eq_deg_km,heading,lon,lat)

    new_plane_lat = np.zeros(9)
    new_plane_lon = np.zeros(9)
    
    for i in xrange(len(plane_lat)):
        new_plane_lat[i] = plane_lat[i]
        new_plane_lon[i] = plane_lon[i]
    
    new_plane_lat[5] = route1[0]
    new_plane_lat[6] = route2[0]
    new_plane_lat[7] = route3[0]
    new_plane_lat[8] = route4[0]
    new_plane_lon[5] = route1[1]
    new_plane_lon[6] = route2[1]
    new_plane_lon[7] = route3[1]
    new_plane_lon[8] = route4[1]
    
    return new_plane_lat,new_plane_lon

################################################################################################################################################################################################                

"""
a function which given a list of discrete probabilities for each destination point, 
it will choose one of those points.

heading_init -- initial direction was headed at last known point
lon_init,lat_init -- last known point of plane in longitude and latitude
km_hop -- how far the plane went in the time interval, 1 hr. So in simplest case, the 777's cruising speed/hr.
k -- affects the heading distribution, based on a Wrapped Cauchy distribution from the current heading (0 deg).
ping_percent_err -- what % error you assume in the Inmarsat 5th ping. either 2.5 or 5%.

uses Wrapped Cauchy distribution for heading

"""
def five_hop_model_wrapped_cauchy(heading_init,lon_init,lat_init,km_hop,k,ping_percent_err):   
    
    #initialize
    plane_lat = np.zeros(5) #initialize plane location after each hop (assumed to be 1 hr for now)
    plane_lon = np.zeros(5)  
    lat = lat_init
    lon = lon_init
    heading = heading_init
        
    for i in xrange(len(plane_lat)):
        new_circle,new_weights,new_angles = wrapped_cauchy_prob_step(heading,k,lon,lat,km_hop/eq_deg_km)
        #new_circle gives up possible coords for diff headings
        
        raw_weights = np.zeros(len(new_circle))
        for j in xrange(len(new_circle)):
            raw_weights[j] = new_weights[j]*ping_prob_normal(inmarsat[0],inmarsat[1],new_circle[j][0],new_circle[j][1],ping_percent_err,dist_from_sat,earth_radius) 
        
        probs = raw_weights / np.sum(raw_weights) #normalize
        
        index = range(len(new_circle))
        chosen = np.random.choice(index,size=None,p=probs)
        #print "chosen",chosen
        
        heading = new_angles[chosen] #update heading
        
        plane_lat[i],plane_lon[i] = new_circle[chosen] #update position
        lat = plane_lat[i]
        lon = plane_lon[i]
    
    #at end of simulation, run the last location & heading for plane for 4 different times
    route1 = make_vector(0.25*km_hop/eq_deg_km,heading,lon,lat)
    route2 = make_vector(0.5*km_hop/eq_deg_km,heading,lon,lat)
    route3 = make_vector(0.75*km_hop/eq_deg_km,heading,lon,lat)
    route4 = make_vector((59./60.)*km_hop/eq_deg_km,heading,lon,lat)

    new_plane_lat = np.zeros(9)
    new_plane_lon = np.zeros(9)
    
    for i in xrange(len(plane_lat)):
        new_plane_lat[i] = plane_lat[i]
        new_plane_lon[i] = plane_lon[i]
    
    new_plane_lat[5] = route1[0]
    new_plane_lat[6] = route2[0]
    new_plane_lat[7] = route3[0]
    new_plane_lat[8] = route4[0]
    new_plane_lon[5] = route1[1]
    new_plane_lon[6] = route2[1]
    new_plane_lon[7] = route3[1]
    new_plane_lon[8] = route4[1]
    
    return new_plane_lat,new_plane_lon

################################################################################################################################################################################################                

last_known_heading = 255.136

km_hop = 905 #assuming 1 hr intervals, at 905 km/hr which is 777 cruising speed -- use for test case
             # max speed of a Boeing 777 is 950 km/hr FYI

N = 1000 #define number of simulations to run -- set to 1000, but set low number to make sure it's working first.

################################################################################################################################################################################################                

if options.km_hop:
    km_hop = float(options.km_hop)
    
if options.last_known_heading:
    last_known_heading = float(options.last_known_heading)

if options.simulations:
    N = int(options.simulations)

################################################################################################################################################################################################                

percenterror1, percenterror2 = 0.05, 0.025

k1, k2, k3 = 0, 1, 10 #Von Mises constants -- uniform, bell, point-like

################################################################################################################################################################################################                

#5 percent error w/ different ks
plane_hops_5per_k1 = []
plane_hops_5per_k2 = []
plane_hops_5per_k3 = []

#2.5 percent error w/ different ks
plane_hops_2per_k1 = []
plane_hops_2per_k2 = []
plane_hops_2per_k3 = []

for i in xrange(N):
    #5% error runs
    plane_hops_5per_k1.append(five_hop_model_von_mises(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k1,percenterror1))
    plane_hops_5per_k2.append(five_hop_model_von_mises(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k2,percenterror1))
    plane_hops_5per_k3.append(five_hop_model_von_mises(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k3,percenterror1))
    #2.5% error runs
    plane_hops_2per_k1.append(five_hop_model_von_mises(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k1,percenterror2))
    plane_hops_2per_k2.append(five_hop_model_von_mises(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k2,percenterror2))
    plane_hops_2per_k3.append(five_hop_model_von_mises(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k3,percenterror2))


################################################################################################################################################################################################                

# 5per_k1 run

first_lat_5per_k1 = []
two_lat_5per_k1 = []
three_lat_5per_k1 = []
four_lat_5per_k1 = []
final_lat_5per_k1 = []

first_lon_5per_k1 = []
two_lon_5per_k1 = []
three_lon_5per_k1 = []
four_lon_5per_k1 = []
final_lon_5per_k1 = []

route1_lat_5per_k1 = []
route2_lat_5per_k1 = []
route3_lat_5per_k1 = []
route4_lat_5per_k1 = []

route1_lon_5per_k1 = []
route2_lon_5per_k1 = []
route3_lon_5per_k1 = []
route4_lon_5per_k1 = []

for i in xrange(len(plane_hops_5per_k1)):
    first_lat_5per_k1.append(plane_hops_5per_k1[i][0][0])
    first_lon_5per_k1.append(plane_hops_5per_k1[i][1][0])
    two_lat_5per_k1.append(plane_hops_5per_k1[i][0][1])
    two_lon_5per_k1.append(plane_hops_5per_k1[i][1][1])
    three_lat_5per_k1.append(plane_hops_5per_k1[i][0][2])
    three_lon_5per_k1.append(plane_hops_5per_k1[i][1][2])
    four_lat_5per_k1.append(plane_hops_5per_k1[i][0][3])
    four_lon_5per_k1.append(plane_hops_5per_k1[i][1][3])
    final_lat_5per_k1.append(plane_hops_5per_k1[i][0][4])
    final_lon_5per_k1.append(plane_hops_5per_k1[i][1][4])
    
    route1_lat_5per_k1.append(plane_hops_5per_k1[i][0][5])
    route1_lon_5per_k1.append(plane_hops_5per_k1[i][1][5])
    route2_lat_5per_k1.append(plane_hops_5per_k1[i][0][6])
    route2_lon_5per_k1.append(plane_hops_5per_k1[i][1][6])
    route3_lat_5per_k1.append(plane_hops_5per_k1[i][0][7])
    route3_lon_5per_k1.append(plane_hops_5per_k1[i][1][7])
    route4_lat_5per_k1.append(plane_hops_5per_k1[i][0][8])
    route4_lon_5per_k1.append(plane_hops_5per_k1[i][1][8])

################################################################################################################################################################################################                

# 5per_k2 run

first_lat_5per_k2 = []
two_lat_5per_k2 = []
three_lat_5per_k2 = []
four_lat_5per_k2 = []
final_lat_5per_k2 = []

first_lon_5per_k2 = []
two_lon_5per_k2 = []
three_lon_5per_k2 = []
four_lon_5per_k2 = []
final_lon_5per_k2 = []

route1_lat_5per_k2 = []
route2_lat_5per_k2 = []
route3_lat_5per_k2 = []
route4_lat_5per_k2 = []

route1_lon_5per_k2 = []
route2_lon_5per_k2 = []
route3_lon_5per_k2 = []
route4_lon_5per_k2 = []

for i in xrange(len(plane_hops_5per_k2)):
    first_lat_5per_k2.append(plane_hops_5per_k2[i][0][0])
    first_lon_5per_k2.append(plane_hops_5per_k2[i][1][0])
    two_lat_5per_k2.append(plane_hops_5per_k2[i][0][1])
    two_lon_5per_k2.append(plane_hops_5per_k2[i][1][1])
    three_lat_5per_k2.append(plane_hops_5per_k2[i][0][2])
    three_lon_5per_k2.append(plane_hops_5per_k2[i][1][2])
    four_lat_5per_k2.append(plane_hops_5per_k2[i][0][3])
    four_lon_5per_k2.append(plane_hops_5per_k2[i][1][3])
    final_lat_5per_k2.append(plane_hops_5per_k2[i][0][4])
    final_lon_5per_k2.append(plane_hops_5per_k2[i][1][4])
    
    route1_lat_5per_k2.append(plane_hops_5per_k2[i][0][5])
    route1_lon_5per_k2.append(plane_hops_5per_k2[i][1][5])
    route2_lat_5per_k2.append(plane_hops_5per_k2[i][0][6])
    route2_lon_5per_k2.append(plane_hops_5per_k2[i][1][6])
    route3_lat_5per_k2.append(plane_hops_5per_k2[i][0][7])
    route3_lon_5per_k2.append(plane_hops_5per_k2[i][1][7])
    route4_lat_5per_k2.append(plane_hops_5per_k2[i][0][8])
    route4_lon_5per_k2.append(plane_hops_5per_k2[i][1][8])

################################################################################################################################################################################################                

# 5per_k3 run

first_lat_5per_k3 = []
two_lat_5per_k3 = []
three_lat_5per_k3 = []
four_lat_5per_k3 = []
final_lat_5per_k3 = []

first_lon_5per_k3 = []
two_lon_5per_k3 = []
three_lon_5per_k3 = []
four_lon_5per_k3 = []
final_lon_5per_k3 = []

route1_lat_5per_k3 = []
route2_lat_5per_k3 = []
route3_lat_5per_k3 = []
route4_lat_5per_k3 = []

route1_lon_5per_k3 = []
route2_lon_5per_k3 = []
route3_lon_5per_k3 = []
route4_lon_5per_k3 = []

for i in xrange(len(plane_hops_5per_k3)):
    first_lat_5per_k3.append(plane_hops_5per_k3[i][0][0])
    first_lon_5per_k3.append(plane_hops_5per_k3[i][1][0])
    two_lat_5per_k3.append(plane_hops_5per_k3[i][0][1])
    two_lon_5per_k3.append(plane_hops_5per_k3[i][1][1])
    three_lat_5per_k3.append(plane_hops_5per_k3[i][0][2])
    three_lon_5per_k3.append(plane_hops_5per_k3[i][1][2])
    four_lat_5per_k3.append(plane_hops_5per_k3[i][0][3])
    four_lon_5per_k3.append(plane_hops_5per_k3[i][1][3])
    final_lat_5per_k3.append(plane_hops_5per_k3[i][0][4])
    final_lon_5per_k3.append(plane_hops_5per_k3[i][1][4])
    
    route1_lat_5per_k3.append(plane_hops_5per_k3[i][0][5])
    route1_lon_5per_k3.append(plane_hops_5per_k3[i][1][5])
    route2_lat_5per_k3.append(plane_hops_5per_k3[i][0][6])
    route2_lon_5per_k3.append(plane_hops_5per_k3[i][1][6])
    route3_lat_5per_k3.append(plane_hops_5per_k3[i][0][7])
    route3_lon_5per_k3.append(plane_hops_5per_k3[i][1][7])
    route4_lat_5per_k3.append(plane_hops_5per_k3[i][0][8])
    route4_lon_5per_k3.append(plane_hops_5per_k3[i][1][8])

################################################################################################################################################################################################                

# 2per_k1 run

first_lat_2per_k1 = []
two_lat_2per_k1 = []
three_lat_2per_k1 = []
four_lat_2per_k1 = []
final_lat_2per_k1 = []

first_lon_2per_k1 = []
two_lon_2per_k1 = []
three_lon_2per_k1 = []
four_lon_2per_k1 = []
final_lon_2per_k1 = []

route1_lat_2per_k1 = []
route2_lat_2per_k1 = []
route3_lat_2per_k1 = []
route4_lat_2per_k1 = []

route1_lon_2per_k1 = []
route2_lon_2per_k1 = []
route3_lon_2per_k1 = []
route4_lon_2per_k1 = []

for i in xrange(len(plane_hops_2per_k1)):
    first_lat_2per_k1.append(plane_hops_2per_k1[i][0][0])
    first_lon_2per_k1.append(plane_hops_2per_k1[i][1][0])
    two_lat_2per_k1.append(plane_hops_2per_k1[i][0][1])
    two_lon_2per_k1.append(plane_hops_2per_k1[i][1][1])
    three_lat_2per_k1.append(plane_hops_2per_k1[i][0][2])
    three_lon_2per_k1.append(plane_hops_2per_k1[i][1][2])
    four_lat_2per_k1.append(plane_hops_2per_k1[i][0][3])
    four_lon_2per_k1.append(plane_hops_2per_k1[i][1][3])
    final_lat_2per_k1.append(plane_hops_2per_k1[i][0][4])
    final_lon_2per_k1.append(plane_hops_2per_k1[i][1][4])
    
    route1_lat_2per_k1.append(plane_hops_2per_k1[i][0][5])
    route1_lon_2per_k1.append(plane_hops_2per_k1[i][1][5])
    route2_lat_2per_k1.append(plane_hops_2per_k1[i][0][6])
    route2_lon_2per_k1.append(plane_hops_2per_k1[i][1][6])
    route3_lat_2per_k1.append(plane_hops_2per_k1[i][0][7])
    route3_lon_2per_k1.append(plane_hops_2per_k1[i][1][7])
    route4_lat_2per_k1.append(plane_hops_2per_k1[i][0][8])
    route4_lon_2per_k1.append(plane_hops_2per_k1[i][1][8])

################################################################################################################################################################################################                

# 2per_k2 run

first_lat_2per_k2 = []
two_lat_2per_k2 = []
three_lat_2per_k2 = []
four_lat_2per_k2 = []
final_lat_2per_k2 = []

first_lon_2per_k2 = []
two_lon_2per_k2 = []
three_lon_2per_k2 = []
four_lon_2per_k2 = []
final_lon_2per_k2 = []

route1_lat_2per_k2 = []
route2_lat_2per_k2 = []
route3_lat_2per_k2 = []
route4_lat_2per_k2 = []

route1_lon_2per_k2 = []
route2_lon_2per_k2 = []
route3_lon_2per_k2 = []
route4_lon_2per_k2 = []

for i in xrange(len(plane_hops_2per_k2)):
    first_lat_2per_k2.append(plane_hops_2per_k2[i][0][0])
    first_lon_2per_k2.append(plane_hops_2per_k2[i][1][0])
    two_lat_2per_k2.append(plane_hops_2per_k2[i][0][1])
    two_lon_2per_k2.append(plane_hops_2per_k2[i][1][1])
    three_lat_2per_k2.append(plane_hops_2per_k2[i][0][2])
    three_lon_2per_k2.append(plane_hops_2per_k2[i][1][2])
    four_lat_2per_k2.append(plane_hops_2per_k2[i][0][3])
    four_lon_2per_k2.append(plane_hops_2per_k2[i][1][3])
    final_lat_2per_k2.append(plane_hops_2per_k2[i][0][4])
    final_lon_2per_k2.append(plane_hops_2per_k2[i][1][4])
    
    route1_lat_2per_k2.append(plane_hops_2per_k2[i][0][5])
    route1_lon_2per_k2.append(plane_hops_2per_k2[i][1][5])
    route2_lat_2per_k2.append(plane_hops_2per_k2[i][0][6])
    route2_lon_2per_k2.append(plane_hops_2per_k2[i][1][6])
    route3_lat_2per_k2.append(plane_hops_2per_k2[i][0][7])
    route3_lon_2per_k2.append(plane_hops_2per_k2[i][1][7])
    route4_lat_2per_k2.append(plane_hops_2per_k2[i][0][8])
    route4_lon_2per_k2.append(plane_hops_2per_k2[i][1][8])

################################################################################################################################################################################################                

# 2per_k3 run

first_lat_2per_k3 = []
two_lat_2per_k3 = []
three_lat_2per_k3 = []
four_lat_2per_k3 = []
final_lat_2per_k3 = []

first_lon_2per_k3 = []
two_lon_2per_k3 = []
three_lon_2per_k3 = []
four_lon_2per_k3 = []
final_lon_2per_k3 = []

route1_lat_2per_k3 = []
route2_lat_2per_k3 = []
route3_lat_2per_k3 = []
route4_lat_2per_k3 = []

route1_lon_2per_k3 = []
route2_lon_2per_k3 = []
route3_lon_2per_k3 = []
route4_lon_2per_k3 = []

for i in xrange(len(plane_hops_2per_k3)):
    first_lat_2per_k3.append(plane_hops_2per_k3[i][0][0])
    first_lon_2per_k3.append(plane_hops_2per_k3[i][1][0])
    two_lat_2per_k3.append(plane_hops_2per_k3[i][0][1])
    two_lon_2per_k3.append(plane_hops_2per_k3[i][1][1])
    three_lat_2per_k3.append(plane_hops_2per_k3[i][0][2])
    three_lon_2per_k3.append(plane_hops_2per_k3[i][1][2])
    four_lat_2per_k3.append(plane_hops_2per_k3[i][0][3])
    four_lon_2per_k3.append(plane_hops_2per_k3[i][1][3])
    final_lat_2per_k3.append(plane_hops_2per_k3[i][0][4])
    final_lon_2per_k3.append(plane_hops_2per_k3[i][1][4])
    
    route1_lat_2per_k3.append(plane_hops_2per_k3[i][0][5])
    route1_lon_2per_k3.append(plane_hops_2per_k3[i][1][5])
    route2_lat_2per_k3.append(plane_hops_2per_k3[i][0][6])
    route2_lon_2per_k3.append(plane_hops_2per_k3[i][1][6])
    route3_lat_2per_k3.append(plane_hops_2per_k3[i][0][7])
    route3_lon_2per_k3.append(plane_hops_2per_k3[i][1][7])
    route4_lat_2per_k3.append(plane_hops_2per_k3[i][0][8])
    route4_lon_2per_k3.append(plane_hops_2per_k3[i][1][8])

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k1,first_lat_5per_k1)
x10,y10 = fig(two_lon_5per_k1,two_lat_5per_k1)
x11,y11 = fig(three_lon_5per_k1,three_lat_5per_k1)
x12,y12 = fig(four_lon_5per_k1,four_lat_5per_k1)
x13,y13 = fig(final_lon_5per_k1,final_lat_5per_k1)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k1,route1_lat_5per_k1)
x15,y15 = fig(route2_lon_5per_k1,route2_lat_5per_k1)
x16,y16 = fig(route3_lon_5per_k1,route3_lat_5per_k1)
x17,y17 = fig(route4_lon_5per_k1,route4_lat_5per_k1)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln69.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k2,first_lat_5per_k2)
x10,y10 = fig(two_lon_5per_k2,two_lat_5per_k2)
x11,y11 = fig(three_lon_5per_k2,three_lat_5per_k2)
x12,y12 = fig(four_lon_5per_k2,four_lat_5per_k2)
x13,y13 = fig(final_lon_5per_k2,final_lat_5per_k2)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k2,route1_lat_5per_k2)
x15,y15 = fig(route2_lon_5per_k2,route2_lat_5per_k2)
x16,y16 = fig(route3_lon_5per_k2,route3_lat_5per_k2)
x17,y17 = fig(route4_lon_5per_k2,route4_lat_5per_k2)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln70.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k3,first_lat_5per_k3)
x10,y10 = fig(two_lon_5per_k3,two_lat_5per_k3)
x11,y11 = fig(three_lon_5per_k3,three_lat_5per_k3)
x12,y12 = fig(four_lon_5per_k3,four_lat_5per_k3)
x13,y13 = fig(final_lon_5per_k3,final_lat_5per_k3)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k3,route1_lat_5per_k3)
x15,y15 = fig(route2_lon_5per_k3,route2_lat_5per_k3)
x16,y16 = fig(route3_lon_5per_k3,route3_lat_5per_k3)
x17,y17 = fig(route4_lon_5per_k3,route4_lat_5per_k3)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln71.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k1,first_lat_5per_k1)
x10,y10 = fig(two_lon_5per_k1,two_lat_5per_k1)
x11,y11 = fig(three_lon_5per_k1,three_lat_5per_k1)
x12,y12 = fig(four_lon_5per_k1,four_lat_5per_k1)
x13,y13 = fig(final_lon_5per_k1,final_lat_5per_k1)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k1,route1_lat_5per_k1)
x15,y15 = fig(route2_lon_5per_k1,route2_lat_5per_k1)
x16,y16 = fig(route3_lon_5per_k1,route3_lat_5per_k1)
x17,y17 = fig(route4_lon_5per_k1,route4_lat_5per_k1)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln72.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k2,first_lat_5per_k2)
x10,y10 = fig(two_lon_5per_k2,two_lat_5per_k2)
x11,y11 = fig(three_lon_5per_k2,three_lat_5per_k2)
x12,y12 = fig(four_lon_5per_k2,four_lat_5per_k2)
x13,y13 = fig(final_lon_5per_k2,final_lat_5per_k2)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k2,route1_lat_5per_k2)
x15,y15 = fig(route2_lon_5per_k2,route2_lat_5per_k2)
x16,y16 = fig(route3_lon_5per_k2,route3_lat_5per_k2)
x17,y17 = fig(route4_lon_5per_k2,route4_lat_5per_k2)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln73.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k3,first_lat_5per_k3)
x10,y10 = fig(two_lon_5per_k3,two_lat_5per_k3)
x11,y11 = fig(three_lon_5per_k3,three_lat_5per_k3)
x12,y12 = fig(four_lon_5per_k3,four_lat_5per_k3)
x13,y13 = fig(final_lon_5per_k3,final_lat_5per_k3)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k3,route1_lat_5per_k3)
x15,y15 = fig(route2_lon_5per_k3,route2_lat_5per_k3)
x16,y16 = fig(route3_lon_5per_k3,route3_lat_5per_k3)
x17,y17 = fig(route4_lon_5per_k3,route4_lat_5per_k3)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln74.svg'
savefig(plt, file)

################################################################################################################################################################################################                

percenterror1, percenterror2 = 0.05, 0.025

k1, k2, k3 = 0, 0.5, 0.99 #Wrapped Cauchy constants -- uniform, bell, point-like

################################################################################################################################################################################################                

#5 percent error w/ different ks
plane_hops_5per_k1 = []
plane_hops_5per_k2 = []
plane_hops_5per_k3 = []

#2.5 percent error w/ different ks
plane_hops_2per_k1 = []
plane_hops_2per_k2 = []
plane_hops_2per_k3 = []

for i in xrange(N):
    #5% error runs
    plane_hops_5per_k1.append(five_hop_model_wrapped_cauchy(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k1,percenterror1))
    plane_hops_5per_k2.append(five_hop_model_wrapped_cauchy(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k2,percenterror1))
    plane_hops_5per_k3.append(five_hop_model_wrapped_cauchy(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k3,percenterror1))
    #2.5% error runs
    plane_hops_2per_k1.append(five_hop_model_wrapped_cauchy(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k1,percenterror2))
    plane_hops_2per_k2.append(five_hop_model_wrapped_cauchy(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k2,percenterror2))
    plane_hops_2per_k3.append(five_hop_model_wrapped_cauchy(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k3,percenterror2))

################################################################################################################################################################################################                

# 5per_k1 run

first_lat_5per_k1 = []
two_lat_5per_k1 = []
three_lat_5per_k1 = []
four_lat_5per_k1 = []
final_lat_5per_k1 = []

first_lon_5per_k1 = []
two_lon_5per_k1 = []
three_lon_5per_k1 = []
four_lon_5per_k1 = []
final_lon_5per_k1 = []

route1_lat_5per_k1 = []
route2_lat_5per_k1 = []
route3_lat_5per_k1 = []
route4_lat_5per_k1 = []

route1_lon_5per_k1 = []
route2_lon_5per_k1 = []
route3_lon_5per_k1 = []
route4_lon_5per_k1 = []

for i in xrange(len(plane_hops_5per_k1)):
    first_lat_5per_k1.append(plane_hops_5per_k1[i][0][0])
    first_lon_5per_k1.append(plane_hops_5per_k1[i][1][0])
    two_lat_5per_k1.append(plane_hops_5per_k1[i][0][1])
    two_lon_5per_k1.append(plane_hops_5per_k1[i][1][1])
    three_lat_5per_k1.append(plane_hops_5per_k1[i][0][2])
    three_lon_5per_k1.append(plane_hops_5per_k1[i][1][2])
    four_lat_5per_k1.append(plane_hops_5per_k1[i][0][3])
    four_lon_5per_k1.append(plane_hops_5per_k1[i][1][3])
    final_lat_5per_k1.append(plane_hops_5per_k1[i][0][4])
    final_lon_5per_k1.append(plane_hops_5per_k1[i][1][4])
    
    route1_lat_5per_k1.append(plane_hops_5per_k1[i][0][5])
    route1_lon_5per_k1.append(plane_hops_5per_k1[i][1][5])
    route2_lat_5per_k1.append(plane_hops_5per_k1[i][0][6])
    route2_lon_5per_k1.append(plane_hops_5per_k1[i][1][6])
    route3_lat_5per_k1.append(plane_hops_5per_k1[i][0][7])
    route3_lon_5per_k1.append(plane_hops_5per_k1[i][1][7])
    route4_lat_5per_k1.append(plane_hops_5per_k1[i][0][8])
    route4_lon_5per_k1.append(plane_hops_5per_k1[i][1][8])

################################################################################################################################################################################################                

# 5per_k2 run

first_lat_5per_k2 = []
two_lat_5per_k2 = []
three_lat_5per_k2 = []
four_lat_5per_k2 = []
final_lat_5per_k2 = []

first_lon_5per_k2 = []
two_lon_5per_k2 = []
three_lon_5per_k2 = []
four_lon_5per_k2 = []
final_lon_5per_k2 = []

route1_lat_5per_k2 = []
route2_lat_5per_k2 = []
route3_lat_5per_k2 = []
route4_lat_5per_k2 = []

route1_lon_5per_k2 = []
route2_lon_5per_k2 = []
route3_lon_5per_k2 = []
route4_lon_5per_k2 = []

for i in xrange(len(plane_hops_5per_k2)):
    first_lat_5per_k2.append(plane_hops_5per_k2[i][0][0])
    first_lon_5per_k2.append(plane_hops_5per_k2[i][1][0])
    two_lat_5per_k2.append(plane_hops_5per_k2[i][0][1])
    two_lon_5per_k2.append(plane_hops_5per_k2[i][1][1])
    three_lat_5per_k2.append(plane_hops_5per_k2[i][0][2])
    three_lon_5per_k2.append(plane_hops_5per_k2[i][1][2])
    four_lat_5per_k2.append(plane_hops_5per_k2[i][0][3])
    four_lon_5per_k2.append(plane_hops_5per_k2[i][1][3])
    final_lat_5per_k2.append(plane_hops_5per_k2[i][0][4])
    final_lon_5per_k2.append(plane_hops_5per_k2[i][1][4])
    
    route1_lat_5per_k2.append(plane_hops_5per_k2[i][0][5])
    route1_lon_5per_k2.append(plane_hops_5per_k2[i][1][5])
    route2_lat_5per_k2.append(plane_hops_5per_k2[i][0][6])
    route2_lon_5per_k2.append(plane_hops_5per_k2[i][1][6])
    route3_lat_5per_k2.append(plane_hops_5per_k2[i][0][7])
    route3_lon_5per_k2.append(plane_hops_5per_k2[i][1][7])
    route4_lat_5per_k2.append(plane_hops_5per_k2[i][0][8])
    route4_lon_5per_k2.append(plane_hops_5per_k2[i][1][8])

################################################################################################################################################################################################                

# 5per_k3 run

first_lat_5per_k3 = []
two_lat_5per_k3 = []
three_lat_5per_k3 = []
four_lat_5per_k3 = []
final_lat_5per_k3 = []

first_lon_5per_k3 = []
two_lon_5per_k3 = []
three_lon_5per_k3 = []
four_lon_5per_k3 = []
final_lon_5per_k3 = []

route1_lat_5per_k3 = []
route2_lat_5per_k3 = []
route3_lat_5per_k3 = []
route4_lat_5per_k3 = []

route1_lon_5per_k3 = []
route2_lon_5per_k3 = []
route3_lon_5per_k3 = []
route4_lon_5per_k3 = []

for i in xrange(len(plane_hops_5per_k3)):
    first_lat_5per_k3.append(plane_hops_5per_k3[i][0][0])
    first_lon_5per_k3.append(plane_hops_5per_k3[i][1][0])
    two_lat_5per_k3.append(plane_hops_5per_k3[i][0][1])
    two_lon_5per_k3.append(plane_hops_5per_k3[i][1][1])
    three_lat_5per_k3.append(plane_hops_5per_k3[i][0][2])
    three_lon_5per_k3.append(plane_hops_5per_k3[i][1][2])
    four_lat_5per_k3.append(plane_hops_5per_k3[i][0][3])
    four_lon_5per_k3.append(plane_hops_5per_k3[i][1][3])
    final_lat_5per_k3.append(plane_hops_5per_k3[i][0][4])
    final_lon_5per_k3.append(plane_hops_5per_k3[i][1][4])
    
    route1_lat_5per_k3.append(plane_hops_5per_k3[i][0][5])
    route1_lon_5per_k3.append(plane_hops_5per_k3[i][1][5])
    route2_lat_5per_k3.append(plane_hops_5per_k3[i][0][6])
    route2_lon_5per_k3.append(plane_hops_5per_k3[i][1][6])
    route3_lat_5per_k3.append(plane_hops_5per_k3[i][0][7])
    route3_lon_5per_k3.append(plane_hops_5per_k3[i][1][7])
    route4_lat_5per_k3.append(plane_hops_5per_k3[i][0][8])
    route4_lon_5per_k3.append(plane_hops_5per_k3[i][1][8])

################################################################################################################################################################################################                

# 2per_k1 run

first_lat_2per_k1 = []
two_lat_2per_k1 = []
three_lat_2per_k1 = []
four_lat_2per_k1 = []
final_lat_2per_k1 = []

first_lon_2per_k1 = []
two_lon_2per_k1 = []
three_lon_2per_k1 = []
four_lon_2per_k1 = []
final_lon_2per_k1 = []

route1_lat_2per_k1 = []
route2_lat_2per_k1 = []
route3_lat_2per_k1 = []
route4_lat_2per_k1 = []

route1_lon_2per_k1 = []
route2_lon_2per_k1 = []
route3_lon_2per_k1 = []
route4_lon_2per_k1 = []

for i in xrange(len(plane_hops_2per_k1)):
    first_lat_2per_k1.append(plane_hops_2per_k1[i][0][0])
    first_lon_2per_k1.append(plane_hops_2per_k1[i][1][0])
    two_lat_2per_k1.append(plane_hops_2per_k1[i][0][1])
    two_lon_2per_k1.append(plane_hops_2per_k1[i][1][1])
    three_lat_2per_k1.append(plane_hops_2per_k1[i][0][2])
    three_lon_2per_k1.append(plane_hops_2per_k1[i][1][2])
    four_lat_2per_k1.append(plane_hops_2per_k1[i][0][3])
    four_lon_2per_k1.append(plane_hops_2per_k1[i][1][3])
    final_lat_2per_k1.append(plane_hops_2per_k1[i][0][4])
    final_lon_2per_k1.append(plane_hops_2per_k1[i][1][4])
    
    route1_lat_2per_k1.append(plane_hops_2per_k1[i][0][5])
    route1_lon_2per_k1.append(plane_hops_2per_k1[i][1][5])
    route2_lat_2per_k1.append(plane_hops_2per_k1[i][0][6])
    route2_lon_2per_k1.append(plane_hops_2per_k1[i][1][6])
    route3_lat_2per_k1.append(plane_hops_2per_k1[i][0][7])
    route3_lon_2per_k1.append(plane_hops_2per_k1[i][1][7])
    route4_lat_2per_k1.append(plane_hops_2per_k1[i][0][8])
    route4_lon_2per_k1.append(plane_hops_2per_k1[i][1][8])

################################################################################################################################################################################################                

# 2per_k2 run

first_lat_2per_k2 = []
two_lat_2per_k2 = []
three_lat_2per_k2 = []
four_lat_2per_k2 = []
final_lat_2per_k2 = []

first_lon_2per_k2 = []
two_lon_2per_k2 = []
three_lon_2per_k2 = []
four_lon_2per_k2 = []
final_lon_2per_k2 = []

route1_lat_2per_k2 = []
route2_lat_2per_k2 = []
route3_lat_2per_k2 = []
route4_lat_2per_k2 = []

route1_lon_2per_k2 = []
route2_lon_2per_k2 = []
route3_lon_2per_k2 = []
route4_lon_2per_k2 = []

for i in xrange(len(plane_hops_2per_k2)):
    first_lat_2per_k2.append(plane_hops_2per_k2[i][0][0])
    first_lon_2per_k2.append(plane_hops_2per_k2[i][1][0])
    two_lat_2per_k2.append(plane_hops_2per_k2[i][0][1])
    two_lon_2per_k2.append(plane_hops_2per_k2[i][1][1])
    three_lat_2per_k2.append(plane_hops_2per_k2[i][0][2])
    three_lon_2per_k2.append(plane_hops_2per_k2[i][1][2])
    four_lat_2per_k2.append(plane_hops_2per_k2[i][0][3])
    four_lon_2per_k2.append(plane_hops_2per_k2[i][1][3])
    final_lat_2per_k2.append(plane_hops_2per_k2[i][0][4])
    final_lon_2per_k2.append(plane_hops_2per_k2[i][1][4])
    
    route1_lat_2per_k2.append(plane_hops_2per_k2[i][0][5])
    route1_lon_2per_k2.append(plane_hops_2per_k2[i][1][5])
    route2_lat_2per_k2.append(plane_hops_2per_k2[i][0][6])
    route2_lon_2per_k2.append(plane_hops_2per_k2[i][1][6])
    route3_lat_2per_k2.append(plane_hops_2per_k2[i][0][7])
    route3_lon_2per_k2.append(plane_hops_2per_k2[i][1][7])
    route4_lat_2per_k2.append(plane_hops_2per_k2[i][0][8])
    route4_lon_2per_k2.append(plane_hops_2per_k2[i][1][8])

################################################################################################################################################################################################                

# 2per_k3 run

first_lat_2per_k3 = []
two_lat_2per_k3 = []
three_lat_2per_k3 = []
four_lat_2per_k3 = []
final_lat_2per_k3 = []

first_lon_2per_k3 = []
two_lon_2per_k3 = []
three_lon_2per_k3 = []
four_lon_2per_k3 = []
final_lon_2per_k3 = []

route1_lat_2per_k3 = []
route2_lat_2per_k3 = []
route3_lat_2per_k3 = []
route4_lat_2per_k3 = []

route1_lon_2per_k3 = []
route2_lon_2per_k3 = []
route3_lon_2per_k3 = []
route4_lon_2per_k3 = []

for i in xrange(len(plane_hops_2per_k3)):
    first_lat_2per_k3.append(plane_hops_2per_k3[i][0][0])
    first_lon_2per_k3.append(plane_hops_2per_k3[i][1][0])
    two_lat_2per_k3.append(plane_hops_2per_k3[i][0][1])
    two_lon_2per_k3.append(plane_hops_2per_k3[i][1][1])
    three_lat_2per_k3.append(plane_hops_2per_k3[i][0][2])
    three_lon_2per_k3.append(plane_hops_2per_k3[i][1][2])
    four_lat_2per_k3.append(plane_hops_2per_k3[i][0][3])
    four_lon_2per_k3.append(plane_hops_2per_k3[i][1][3])
    final_lat_2per_k3.append(plane_hops_2per_k3[i][0][4])
    final_lon_2per_k3.append(plane_hops_2per_k3[i][1][4])
    
    route1_lat_2per_k3.append(plane_hops_2per_k3[i][0][5])
    route1_lon_2per_k3.append(plane_hops_2per_k3[i][1][5])
    route2_lat_2per_k3.append(plane_hops_2per_k3[i][0][6])
    route2_lon_2per_k3.append(plane_hops_2per_k3[i][1][6])
    route3_lat_2per_k3.append(plane_hops_2per_k3[i][0][7])
    route3_lon_2per_k3.append(plane_hops_2per_k3[i][1][7])
    route4_lat_2per_k3.append(plane_hops_2per_k3[i][0][8])
    route4_lon_2per_k3.append(plane_hops_2per_k3[i][1][8])


################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k1,first_lat_5per_k1)
x10,y10 = fig(two_lon_5per_k1,two_lat_5per_k1)
x11,y11 = fig(three_lon_5per_k1,three_lat_5per_k1)
x12,y12 = fig(four_lon_5per_k1,four_lat_5per_k1)
x13,y13 = fig(final_lon_5per_k1,final_lat_5per_k1)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k1,route1_lat_5per_k1)
x15,y15 = fig(route2_lon_5per_k1,route2_lat_5per_k1)
x16,y16 = fig(route3_lon_5per_k1,route3_lat_5per_k1)
x17,y17 = fig(route4_lon_5per_k1,route4_lat_5per_k1)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln102.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k2,first_lat_5per_k2)
x10,y10 = fig(two_lon_5per_k2,two_lat_5per_k2)
x11,y11 = fig(three_lon_5per_k2,three_lat_5per_k2)
x12,y12 = fig(four_lon_5per_k2,four_lat_5per_k2)
x13,y13 = fig(final_lon_5per_k2,final_lat_5per_k2)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k2,route1_lat_5per_k2)
x15,y15 = fig(route2_lon_5per_k2,route2_lat_5per_k2)
x16,y16 = fig(route3_lon_5per_k2,route3_lat_5per_k2)
x17,y17 = fig(route4_lon_5per_k2,route4_lat_5per_k2)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln103.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
x7,y7 = fig(circle_lon_err2_5per,circle_lat_err2_5per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k3,first_lat_5per_k3)
x10,y10 = fig(two_lon_5per_k3,two_lat_5per_k3)
x11,y11 = fig(three_lon_5per_k3,three_lat_5per_k3)
x12,y12 = fig(four_lon_5per_k3,four_lat_5per_k3)
x13,y13 = fig(final_lon_5per_k3,final_lat_5per_k3)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k3,route1_lat_5per_k3)
x15,y15 = fig(route2_lon_5per_k3,route2_lat_5per_k3)
x16,y16 = fig(route3_lon_5per_k3,route3_lat_5per_k3)
x17,y17 = fig(route4_lon_5per_k3,route4_lat_5per_k3)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln104.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])


#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k1,first_lat_5per_k1)
x10,y10 = fig(two_lon_5per_k1,two_lat_5per_k1)
x11,y11 = fig(three_lon_5per_k1,three_lat_5per_k1)
x12,y12 = fig(four_lon_5per_k1,four_lat_5per_k1)
x13,y13 = fig(final_lon_5per_k1,final_lat_5per_k1)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k1,route1_lat_5per_k1)
x15,y15 = fig(route2_lon_5per_k1,route2_lat_5per_k1)
x16,y16 = fig(route3_lon_5per_k1,route3_lat_5per_k1)
x17,y17 = fig(route4_lon_5per_k1,route4_lat_5per_k1)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln105.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k2,first_lat_5per_k2)
x10,y10 = fig(two_lon_5per_k2,two_lat_5per_k2)
x11,y11 = fig(three_lon_5per_k2,three_lat_5per_k2)
x12,y12 = fig(four_lon_5per_k2,four_lat_5per_k2)
x13,y13 = fig(final_lon_5per_k2,final_lat_5per_k2)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k2,route1_lat_5per_k2)
x15,y15 = fig(route2_lon_5per_k2,route2_lat_5per_k2)
x16,y16 = fig(route3_lon_5per_k2,route3_lat_5per_k2)
x17,y17 = fig(route4_lon_5per_k2,route4_lat_5per_k2)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln106.svg'
savefig(plt, file)


################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- 2.5% error
x5,y5 = fig(circle_lon,circle_lat)
x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)

#Add points after each hr
x9,y9 = fig(first_lon_5per_k3,first_lat_5per_k3)
x10,y10 = fig(two_lon_5per_k3,two_lat_5per_k3)
x11,y11 = fig(three_lon_5per_k3,three_lat_5per_k3)
x12,y12 = fig(four_lon_5per_k3,four_lat_5per_k3)
x13,y13 = fig(final_lon_5per_k3,final_lat_5per_k3)

#Add ultimate locations of MH370
x14,y14 = fig(route1_lon_5per_k3,route1_lat_5per_k3)
x15,y15 = fig(route2_lon_5per_k3,route2_lat_5per_k3)
x16,y16 = fig(route3_lon_5per_k3,route3_lat_5per_k3)
x17,y17 = fig(route4_lon_5per_k3,route4_lat_5per_k3)

#Draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5% error')
fig.plot(x7,y7,'r--',markersize=5)

#Plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Add monte carlo points
fig.plot(x9,y9,'yo',markersize=5,label='after 1 hr')
fig.plot(x10,y10,'co',markersize=5,label='after 2 hrs')
fig.plot(x11,y11,'mo',markersize=5,label= 'after 3 hrs')
fig.plot(x12,y12,'wo',markersize=5,label='after 4 hrs')
fig.plot(x13,y13,'ro',markersize=7,label='after 5 hrs')

#Plot ultimate locations of MH370
fig.plot(x14,y14,'bo',markersize=5,label='in final hr')
fig.plot(x15,y15,'bo',markersize=5)
fig.plot(x16,y16,'bo',markersize=5)
fig.plot(x17,y17,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Position Progression Over Time', fontsize=30)

#Show below
#plt.show()

file = 'part2_plot_ln107.svg'
savefig(plt, file)

################################################################################################################################################################################################                
# PART 3
################################################################################################################################################################################################                

#add other ping distances, and then replot

ping_distances = np.array([4036.99, 4194.65, 4352.32, 4509.99, 4667.65, 4825.32])
ping_times = np.array([0.9333, 1, 1, 1, 1, 1]) #make 1st hop slightly smaller owing to time difference
ping_arcs = np.array([34.8485, 36.2649, 37.6812, 39.0976, 40.5139, 41.9303, 43.3466])

################################################################################################################################################################################################                

#make points for 6 circles -- opt not to use for loop
ping_circle_211am = make_circle(ping_arcs[0],360,64.5,0)
ping_circle_311am = make_circle(ping_arcs[1],360,64.5,0)
ping_circle_411am = make_circle(ping_arcs[2],360,64.5,0)
ping_circle_511am = make_circle(ping_arcs[3],360,64.5,0)
ping_circle_611am = make_circle(ping_arcs[4],360,64.5,0)
ping_circle_711am = make_circle(ping_arcs[5],360,64.5,0)
ping_circle_811am = make_circle(ping_arcs[6],360,64.5,0)

#initialize lat & lon lists
circle_lon_211am = []
circle_lat_211am = []
circle_lat_311am = []
circle_lon_311am = []
circle_lat_411am = []
circle_lon_411am = []
circle_lat_511am = []
circle_lon_511am = []
circle_lat_611am = []
circle_lon_611am = []
circle_lat_711am = []
circle_lon_711am = []
circle_lat_811am = []
circle_lon_811am = []

for i in xrange(len(ping_circle_211am)): #they're all the same length so just do it once
    circle_lat_211am.append(ping_circle_211am[i][0])
    circle_lon_211am.append(ping_circle_211am[i][1])
    
    circle_lat_311am.append(ping_circle_311am[i][0])
    circle_lon_311am.append(ping_circle_311am[i][1])

    circle_lat_411am.append(ping_circle_411am[i][0])
    circle_lon_411am.append(ping_circle_411am[i][1])

    circle_lat_511am.append(ping_circle_511am[i][0])
    circle_lon_511am.append(ping_circle_511am[i][1])

    circle_lat_611am.append(ping_circle_611am[i][0])
    circle_lon_611am.append(ping_circle_611am[i][1])

    circle_lat_711am.append(ping_circle_711am[i][0])
    circle_lon_711am.append(ping_circle_711am[i][1])

    circle_lat_811am.append(ping_circle_811am[i][0])
    circle_lon_811am.append(ping_circle_811am[i][1])

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
x5,y5 = fig(circle_lon_211am,circle_lat_211am)
x6,y6 = fig(circle_lon_311am,circle_lat_311am)
x7,y7 = fig(circle_lon_411am,circle_lat_411am)
x8,y8 = fig(circle_lon_511am,circle_lat_511am)
x9,y9 = fig(circle_lon_611am,circle_lat_611am)
x10,y10 = fig(circle_lon_711am,circle_lat_711am)
x11,y11 = fig(circle_lon_811am,circle_lat_811am)

#Draw circle showing extent of Inmarsat sat radar detection for each of the pings
fig.plot(x5,y5,'r--',markersize=5,label='1st Ping Arc')
fig.plot(x6,y6,'r-',markersize=5, label='Ping Arcs After Disappearance')
fig.plot(x7,y7,'r-',markersize=5)
fig.plot(x8,y8,'r-',markersize=5)
fig.plot(x9,y9,'r-',markersize=5)
fig.plot(x10,y10,'r-',markersize=5)
fig.plot(x11,y11,'r-',markersize=5)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Individual Pings', fontsize=30)

#Show below
#plt.show()

file = 'part3_plot_ln68.svg'
savefig(plt, file)

################################################################################################################################################################################################                

"""
a function which given a list of discrete probabilities for each destination point, 
it will choose one of those points.

heading_init -- initial direction was headed at last known point
lon_init,lat_init -- last known point of plane in longitude and latitude
km_hop -- how far the plane went in the time interval, 1 hr. So in simplest case, the 777's cruising speed/hr.
std_dev -- the standard deviation of the heading, based on a normal distribution from the current heading (0 deg).
ping_percent_err -- what % error you assume in the Inmarsat 5th ping. either 2.5 or 5%.

uses normal distribution for heading

replace "dist_from_sat" with "ping_distance" since that's changing. run 6 times.

"""
def six_hop_model_normal(heading_init,lon_init,lat_init,km_hop,std_dev,ping_percent_err,ping_distances,ping_times):   
    
    #initialize
    plane_lat = np.zeros(6) #initialize plane location after each hop (assumed to be 1 hr for now)
    plane_lon = np.zeros(6)  
    lat = lat_init
    lon = lon_init
    heading = heading_init
    
    for i in xrange(len(plane_lat)):
        new_circle,new_weights,new_angles = normal_prob_step(heading,std_dev,lon,lat,(km_hop/eq_deg_km)*ping_times[i])
        #new_circle gives up possible coords for diff headings
        
        raw_weights = np.zeros(len(new_circle))
        for j in xrange(len(new_circle)):
            raw_weights[j] = new_weights[j]*ping_prob_normal(inmarsat[0],inmarsat[1],new_circle[j][0],new_circle[j][1],ping_percent_err,ping_distances[i],earth_radius) 
        
        probs = raw_weights / np.sum(raw_weights) #normalize    
        
        index = range(len(new_circle))
        chosen = np.random.choice(index,size=None,p=probs)
        #print "chosen",chosen
        
        heading = new_angles[chosen] #update heading
        
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
    
    new_plane_lat[6] = route1[0] # add 1 for 6 hops instead of 5
    new_plane_lat[7] = route2[0] # add 1 for 6 hops instead of 5
    new_plane_lat[8] = route3[0] # add 1 for 6 hops instead of 5
    new_plane_lat[9] = route4[0] # add 1 for 6 hops instead of 5
    new_plane_lon[6] = route1[1] # add 1 for 6 hops instead of 5
    new_plane_lon[7] = route2[1] # add 1 for 6 hops instead of 5
    new_plane_lon[8] = route3[1] # add 1 for 6 hops instead of 5
    new_plane_lon[9] = route4[1] # add 1 for 6 hops instead of 5
    
    return new_plane_lat,new_plane_lon

################################################################################################################################################################################################                

"""
a function which given a list of discrete probabilities for each destination point, 
it will choose one of those points.

heading_init -- initial direction was headed at last known point
lon_init,lat_init -- last known point of plane in longitude and latitude
km_hop -- how far the plane went in the time interval, 1 hr. So in simplest case, the 777's cruising speed/hr.
k -- affects the heading distribution, based on a Von Mises distribution from the current heading (0 deg).
ping_percent_err -- what % error you assume in the Inmarsat 5th ping. either 2.5 or 5%.

uses Von Mises distribution for heading

replace "dist_from_sat" with "ping_distance" since that's changing. run 6 times.

"""
def six_hop_model_von_mises(heading_init,lon_init,lat_init,km_hop,k,ping_percent_err,ping_distances,ping_times):   
    
    #initialize
    plane_lat = np.zeros(6) #initialize plane location after each hop (assumed to be 1 hr for now)
    plane_lon = np.zeros(6)  
    lat = lat_init
    lon = lon_init
    heading = heading_init
    
    for i in xrange(len(plane_lat)):
        new_circle,new_weights,new_angles = von_mises_prob_step(heading,k,lon,lat,(km_hop/eq_deg_km)*ping_times[i])
        #new_circle gives up possible coords for diff headings
        
        raw_weights = np.zeros(len(new_circle))
        for j in xrange(len(new_circle)):
            raw_weights[j] = new_weights[j]*ping_prob_normal(inmarsat[0],inmarsat[1],new_circle[j][0],new_circle[j][1],ping_percent_err,ping_distances[i],earth_radius) 
        
        probs = raw_weights / np.sum(raw_weights) #normalize    
        
        index = range(len(new_circle))
        chosen = np.random.choice(index,size=None,p=probs)
        #print "chosen",chosen
        
        heading = new_angles[chosen] #update heading
        
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
    
    new_plane_lat[6] = route1[0] # add 1 for 6 hops instead of 5
    new_plane_lat[7] = route2[0] # add 1 for 6 hops instead of 5
    new_plane_lat[8] = route3[0] # add 1 for 6 hops instead of 5
    new_plane_lat[9] = route4[0] # add 1 for 6 hops instead of 5
    new_plane_lon[6] = route1[1] # add 1 for 6 hops instead of 5
    new_plane_lon[7] = route2[1] # add 1 for 6 hops instead of 5
    new_plane_lon[8] = route3[1] # add 1 for 6 hops instead of 5
    new_plane_lon[9] = route4[1] # add 1 for 6 hops instead of 5
    
    return new_plane_lat,new_plane_lon

################################################################################################################################################################################################                

"""
a function which given a list of discrete probabilities for each destination point, 
it will choose one of those points.

heading_init -- initial direction was headed at last known point
lon_init,lat_init -- last known point of plane in longitude and latitude
km_hop -- how far the plane went in the time interval, 1 hr. So in simplest case, the 777's cruising speed/hr.
k -- affects the heading distribution, based on a Wrapped Cauchy distribution from the current heading (0 deg).
ping_percent_err -- what % error you assume in the Inmarsat 5th ping. either 2.5 or 5%.

uses Wrapped Cauchy distribution for heading

replace "dist_from_sat" with "ping_distance" since that's changing. run 6 times.

"""
def six_hop_model_wrapped_cauchy(heading_init,lon_init,lat_init,km_hop,k,ping_percent_err,ping_distances,ping_times):   
    
    #initialize
    plane_lat = np.zeros(6) #initialize plane location after each hop (assumed to be 1 hr for now)
    plane_lon = np.zeros(6)  
    lat = lat_init
    lon = lon_init
    heading = heading_init
    
    for i in xrange(len(plane_lat)):
        new_circle,new_weights,new_angles = wrapped_cauchy_prob_step(heading,k,lon,lat,(km_hop/eq_deg_km)*ping_times[i])
        #new_circle gives up possible coords for diff headings
        
        raw_weights = np.zeros(len(new_circle))
        for j in xrange(len(new_circle)):
            raw_weights[j] = new_weights[j]*ping_prob_normal(inmarsat[0],inmarsat[1],new_circle[j][0],new_circle[j][1],ping_percent_err,ping_distances[i],earth_radius) 
        
        probs = raw_weights / np.sum(raw_weights) #normalize    
        
        index = range(len(new_circle))
        chosen = np.random.choice(index,size=None,p=probs)
        #print "chosen",chosen
        
        heading = new_angles[chosen] #update heading
        
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
    
    new_plane_lat[6] = route1[0] # add 1 for 6 hops instead of 5
    new_plane_lat[7] = route2[0] # add 1 for 6 hops instead of 5
    new_plane_lat[8] = route3[0] # add 1 for 6 hops instead of 5
    new_plane_lat[9] = route4[0] # add 1 for 6 hops instead of 5
    new_plane_lon[6] = route1[1] # add 1 for 6 hops instead of 5
    new_plane_lon[7] = route2[1] # add 1 for 6 hops instead of 5
    new_plane_lon[8] = route3[1] # add 1 for 6 hops instead of 5
    new_plane_lon[9] = route4[1] # add 1 for 6 hops instead of 5
    
    return new_plane_lat,new_plane_lon

################################################################################################################################################################################################                

last_known_heading = 255.136 #calculated in Mathematica from MH370's two last publically known locations:
                             #when it deviated from its flight path, and when it was last detected by Malyasian military radar
                             #0 degrees is due north, so this is basically to the west (270 degrees), but slightly south

km_hop = 905 #assuming 1 hr intervals, at 905 km/hr which is 777 cruising speed -- use for test case
             # max speed of a Boeing 777 is 950 km/hr FYI

N = 1000 #define number of simulations to run

################################################################################################################################################################################################                

#override simulation params with cli options

if options.km_hop:
    km_hop = float(options.km_hop)
    
if options.last_known_heading:
    last_known_heading = float(options.last_known_heading)

if options.simulations:
    N = int(options.simulations)

################################################################################################################################################################################################                

percenterror1,percenterror2 = 0.05, 0.025

std_dev = 30

################################################################################################################################################################################################                

#override simulation params with cli options

if options.std_dev:
    std_dev = float(options.std_dev)

################################################################################################################################################################################################                


plane_hops_5per = []
plane_hops_2per = []

for i in xrange(N):
    plane_hops_5per.append(six_hop_model_normal(last_known_heading,pulauperak[1],pulauperak[0],km_hop,std_dev,percenterror1,ping_distances,ping_times))
    plane_hops_2per.append(six_hop_model_normal(last_known_heading,pulauperak[1],pulauperak[0],km_hop,std_dev,percenterror2,ping_distances,ping_times))

################################################################################################################################################################################################                

first_lat_5per = []
two_lat_5per = []
three_lat_5per = []
four_lat_5per = []
five_lat_5per = []
final_lat_5per = []

first_lon_5per = []
two_lon_5per = []
three_lon_5per = []
four_lon_5per = []
five_lon_5per = []
final_lon_5per = []

route1_lat_5per = []
route2_lat_5per = []
route3_lat_5per = []
route4_lat_5per = []

route1_lon_5per = []
route2_lon_5per = []
route3_lon_5per = []
route4_lon_5per = []

for i in xrange(len(plane_hops_5per)):
    first_lat_5per.append(plane_hops_5per[i][0][0])
    first_lon_5per.append(plane_hops_5per[i][1][0])
    two_lat_5per.append(plane_hops_5per[i][0][1])
    two_lon_5per.append(plane_hops_5per[i][1][1])
    three_lat_5per.append(plane_hops_5per[i][0][2])
    three_lon_5per.append(plane_hops_5per[i][1][2])
    four_lat_5per.append(plane_hops_5per[i][0][3])
    four_lon_5per.append(plane_hops_5per[i][1][3])
    five_lat_5per.append(plane_hops_5per[i][0][4])
    five_lon_5per.append(plane_hops_5per[i][1][4])
    final_lat_5per.append(plane_hops_5per[i][0][5])
    final_lon_5per.append(plane_hops_5per[i][1][5])
    
    route1_lat_5per.append(plane_hops_5per[i][0][6])
    route1_lon_5per.append(plane_hops_5per[i][1][6])
    route2_lat_5per.append(plane_hops_5per[i][0][7])
    route2_lon_5per.append(plane_hops_5per[i][1][7])
    route3_lat_5per.append(plane_hops_5per[i][0][8])
    route3_lon_5per.append(plane_hops_5per[i][1][8])
    route4_lat_5per.append(plane_hops_5per[i][0][9])
    route4_lon_5per.append(plane_hops_5per[i][1][9])


################################################################################################################################################################################################                

first_lat_2per = []
two_lat_2per = []
three_lat_2per = []
four_lat_2per = []
five_lat_2per = []
final_lat_2per = []

first_lon_2per = []
two_lon_2per = []
three_lon_2per = []
four_lon_2per = []
five_lon_2per = []
final_lon_2per = []

route1_lat_2per = []
route2_lat_2per = []
route3_lat_2per = []
route4_lat_2per = []

route1_lon_2per = []
route2_lon_2per = []
route3_lon_2per = []
route4_lon_2per = []

for i in xrange(len(plane_hops_2per)):
    first_lat_2per.append(plane_hops_2per[i][0][0])
    first_lon_2per.append(plane_hops_2per[i][1][0])
    two_lat_2per.append(plane_hops_2per[i][0][1])
    two_lon_2per.append(plane_hops_2per[i][1][1])
    three_lat_2per.append(plane_hops_2per[i][0][2])
    three_lon_2per.append(plane_hops_2per[i][1][2])
    four_lat_2per.append(plane_hops_2per[i][0][3])
    four_lon_2per.append(plane_hops_2per[i][1][3])
    five_lat_2per.append(plane_hops_2per[i][0][4])
    five_lon_2per.append(plane_hops_2per[i][1][4])
    final_lat_2per.append(plane_hops_2per[i][0][5])
    final_lon_2per.append(plane_hops_2per[i][1][5])
    
    route1_lat_2per.append(plane_hops_2per[i][0][6])
    route1_lon_2per.append(plane_hops_2per[i][1][6])
    route2_lat_2per.append(plane_hops_2per[i][0][7])
    route2_lon_2per.append(plane_hops_2per[i][1][7])
    route3_lat_2per.append(plane_hops_2per[i][0][8])
    route3_lon_2per.append(plane_hops_2per[i][1][8])
    route4_lat_2per.append(plane_hops_2per[i][0][9])
    route4_lon_2per.append(plane_hops_2per[i][1][9])

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
x5,y5 = fig(circle_lon_211am,circle_lat_211am)
x6,y6 = fig(circle_lon_311am,circle_lat_311am)
x7,y7 = fig(circle_lon_411am,circle_lat_411am)
x8,y8 = fig(circle_lon_511am,circle_lat_511am)
x9,y9 = fig(circle_lon_611am,circle_lat_611am)
x10,y10 = fig(circle_lon_711am,circle_lat_711am)
x11,y11 = fig(circle_lon_811am,circle_lat_811am)

#Add points after each hr
x12,y12 = fig(first_lon_5per,first_lat_5per)
x13,y13 = fig(two_lon_5per,two_lat_5per)
x14,y14 = fig(three_lon_5per,three_lat_5per)
x15,y15 = fig(four_lon_5per,four_lat_5per)
x16,y16 = fig(five_lon_5per,five_lat_5per)
x17,y17 = fig(final_lon_5per,final_lat_5per)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_5per,route1_lat_5per)
x19,y19 = fig(route2_lon_5per,route2_lat_5per)
x20,y20 = fig(route3_lon_5per,route3_lat_5per)
x21,y21 = fig(route4_lon_5per,route4_lat_5per)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw circle showing extent of Inmarsat sat radar detection for each of the pings
fig.plot(x5,y5,'r--',markersize=5,label='1st Ping Arc')
fig.plot(x6,y6,'r-',markersize=5, label='Ping Arcs After Disappearance')
fig.plot(x7,y7,'r-',markersize=5)
fig.plot(x8,y8,'r-',markersize=5)
fig.plot(x9,y9,'r-',markersize=5)
fig.plot(x10,y10,'r-',markersize=5)
fig.plot(x11,y11,'r-',markersize=5)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'wo',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'bo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Individual Pings', fontsize=30)

#Show below
#plt.show()

file = 'part3_plot_ln77.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
x5,y5 = fig(circle_lon_211am,circle_lat_211am)
x6,y6 = fig(circle_lon_311am,circle_lat_311am)
x7,y7 = fig(circle_lon_411am,circle_lat_411am)
x8,y8 = fig(circle_lon_511am,circle_lat_511am)
x9,y9 = fig(circle_lon_611am,circle_lat_611am)
x10,y10 = fig(circle_lon_711am,circle_lat_711am)
x11,y11 = fig(circle_lon_811am,circle_lat_811am)

#Add points after each hr
x12,y12 = fig(first_lon_2per,first_lat_2per)
x13,y13 = fig(two_lon_2per,two_lat_2per)
x14,y14 = fig(three_lon_2per,three_lat_2per)
x15,y15 = fig(four_lon_2per,four_lat_2per)
x16,y16 = fig(five_lon_2per,five_lat_2per)
x17,y17 = fig(final_lon_2per,final_lat_2per)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_2per,route1_lat_2per)
x19,y19 = fig(route2_lon_2per,route2_lat_2per)
x20,y20 = fig(route3_lon_2per,route3_lat_2per)
x21,y21 = fig(route4_lon_2per,route4_lat_2per)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw circle showing extent of Inmarsat sat radar detection for each of the pings
fig.plot(x5,y5,'r--',markersize=5,label='1st Ping Arc')
fig.plot(x6,y6,'r-',markersize=5, label='Ping Arcs After Disappearance')
fig.plot(x7,y7,'r-',markersize=5)
fig.plot(x8,y8,'r-',markersize=5)
fig.plot(x9,y9,'r-',markersize=5)
fig.plot(x10,y10,'r-',markersize=5)
fig.plot(x11,y11,'r-',markersize=5)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'wo',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'bo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Individual Pings', fontsize=30)

#Show below
#plt.show()

file = 'part3_plot_ln78.svg'
savefig(plt, file)

################################################################################################################################################################################################                

percenterror1,percenterror2 = 0.05, 0.025

k = 10

################################################################################################################################################################################################                

plane_hops_5per = []
plane_hops_2per = []

for i in xrange(N):
    plane_hops_5per.append(six_hop_model_von_mises(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k,percenterror1,ping_distances,ping_times))
    plane_hops_2per.append(six_hop_model_von_mises(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k,percenterror2,ping_distances,ping_times))

################################################################################################################################################################################################                

first_lat_5per = []
two_lat_5per = []
three_lat_5per = []
four_lat_5per = []
five_lat_5per = []
final_lat_5per = []

first_lon_5per = []
two_lon_5per = []
three_lon_5per = []
four_lon_5per = []
five_lon_5per = []
final_lon_5per = []

route1_lat_5per = []
route2_lat_5per = []
route3_lat_5per = []
route4_lat_5per = []

route1_lon_5per = []
route2_lon_5per = []
route3_lon_5per = []
route4_lon_5per = []

for i in xrange(len(plane_hops_5per)):
    first_lat_5per.append(plane_hops_5per[i][0][0])
    first_lon_5per.append(plane_hops_5per[i][1][0])
    two_lat_5per.append(plane_hops_5per[i][0][1])
    two_lon_5per.append(plane_hops_5per[i][1][1])
    three_lat_5per.append(plane_hops_5per[i][0][2])
    three_lon_5per.append(plane_hops_5per[i][1][2])
    four_lat_5per.append(plane_hops_5per[i][0][3])
    four_lon_5per.append(plane_hops_5per[i][1][3])
    five_lat_5per.append(plane_hops_5per[i][0][4])
    five_lon_5per.append(plane_hops_5per[i][1][4])
    final_lat_5per.append(plane_hops_5per[i][0][5])
    final_lon_5per.append(plane_hops_5per[i][1][5])
    
    route1_lat_5per.append(plane_hops_5per[i][0][6])
    route1_lon_5per.append(plane_hops_5per[i][1][6])
    route2_lat_5per.append(plane_hops_5per[i][0][7])
    route2_lon_5per.append(plane_hops_5per[i][1][7])
    route3_lat_5per.append(plane_hops_5per[i][0][8])
    route3_lon_5per.append(plane_hops_5per[i][1][8])
    route4_lat_5per.append(plane_hops_5per[i][0][9])
    route4_lon_5per.append(plane_hops_5per[i][1][9])

################################################################################################################################################################################################                

first_lat_2per = []
two_lat_2per = []
three_lat_2per = []
four_lat_2per = []
five_lat_2per = []
final_lat_2per = []

first_lon_2per = []
two_lon_2per = []
three_lon_2per = []
four_lon_2per = []
five_lon_2per = []
final_lon_2per = []

route1_lat_2per = []
route2_lat_2per = []
route3_lat_2per = []
route4_lat_2per = []

route1_lon_2per = []
route2_lon_2per = []
route3_lon_2per = []
route4_lon_2per = []

for i in xrange(len(plane_hops_2per)):
    first_lat_2per.append(plane_hops_2per[i][0][0])
    first_lon_2per.append(plane_hops_2per[i][1][0])
    two_lat_2per.append(plane_hops_2per[i][0][1])
    two_lon_2per.append(plane_hops_2per[i][1][1])
    three_lat_2per.append(plane_hops_2per[i][0][2])
    three_lon_2per.append(plane_hops_2per[i][1][2])
    four_lat_2per.append(plane_hops_2per[i][0][3])
    four_lon_2per.append(plane_hops_2per[i][1][3])
    five_lat_2per.append(plane_hops_2per[i][0][4])
    five_lon_2per.append(plane_hops_2per[i][1][4])
    final_lat_2per.append(plane_hops_2per[i][0][5])
    final_lon_2per.append(plane_hops_2per[i][1][5])
    
    route1_lat_2per.append(plane_hops_2per[i][0][6])
    route1_lon_2per.append(plane_hops_2per[i][1][6])
    route2_lat_2per.append(plane_hops_2per[i][0][7])
    route2_lon_2per.append(plane_hops_2per[i][1][7])
    route3_lat_2per.append(plane_hops_2per[i][0][8])
    route3_lon_2per.append(plane_hops_2per[i][1][8])
    route4_lat_2per.append(plane_hops_2per[i][0][9])
    route4_lon_2per.append(plane_hops_2per[i][1][9])

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
x5,y5 = fig(circle_lon_211am,circle_lat_211am)
x6,y6 = fig(circle_lon_311am,circle_lat_311am)
x7,y7 = fig(circle_lon_411am,circle_lat_411am)
x8,y8 = fig(circle_lon_511am,circle_lat_511am)
x9,y9 = fig(circle_lon_611am,circle_lat_611am)
x10,y10 = fig(circle_lon_711am,circle_lat_711am)
x11,y11 = fig(circle_lon_811am,circle_lat_811am)

#Add points after each hr
x12,y12 = fig(first_lon_5per,first_lat_5per)
x13,y13 = fig(two_lon_5per,two_lat_5per)
x14,y14 = fig(three_lon_5per,three_lat_5per)
x15,y15 = fig(four_lon_5per,four_lat_5per)
x16,y16 = fig(five_lon_5per,five_lat_5per)
x17,y17 = fig(final_lon_5per,final_lat_5per)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_5per,route1_lat_5per)
x19,y19 = fig(route2_lon_5per,route2_lat_5per)
x20,y20 = fig(route3_lon_5per,route3_lat_5per)
x21,y21 = fig(route4_lon_5per,route4_lat_5per)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw circle showing extent of Inmarsat sat radar detection for each of the pings
fig.plot(x5,y5,'r--',markersize=5,label='1st Ping Arc')
fig.plot(x6,y6,'r-',markersize=5, label='Ping Arcs After Disappearance')
fig.plot(x7,y7,'r-',markersize=5)
fig.plot(x8,y8,'r-',markersize=5)
fig.plot(x9,y9,'r-',markersize=5)
fig.plot(x10,y10,'r-',markersize=5)
fig.plot(x11,y11,'r-',markersize=5)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'wo',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'bo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Individual Pings', fontsize=30)

#Show below
#plt.show()

file = 'part3_plot_ln83.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
x5,y5 = fig(circle_lon_211am,circle_lat_211am)
x6,y6 = fig(circle_lon_311am,circle_lat_311am)
x7,y7 = fig(circle_lon_411am,circle_lat_411am)
x8,y8 = fig(circle_lon_511am,circle_lat_511am)
x9,y9 = fig(circle_lon_611am,circle_lat_611am)
x10,y10 = fig(circle_lon_711am,circle_lat_711am)
x11,y11 = fig(circle_lon_811am,circle_lat_811am)

#Add points after each hr
x12,y12 = fig(first_lon_2per,first_lat_2per)
x13,y13 = fig(two_lon_2per,two_lat_2per)
x14,y14 = fig(three_lon_2per,three_lat_2per)
x15,y15 = fig(four_lon_2per,four_lat_2per)
x16,y16 = fig(five_lon_2per,five_lat_2per)
x17,y17 = fig(final_lon_2per,final_lat_2per)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_2per,route1_lat_2per)
x19,y19 = fig(route2_lon_2per,route2_lat_2per)
x20,y20 = fig(route3_lon_2per,route3_lat_2per)
x21,y21 = fig(route4_lon_2per,route4_lat_2per)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw circle showing extent of Inmarsat sat radar detection for each of the pings
fig.plot(x5,y5,'r--',markersize=5,label='1st Ping Arc')
fig.plot(x6,y6,'r-',markersize=5, label='Ping Arcs After Disappearance')
fig.plot(x7,y7,'r-',markersize=5)
fig.plot(x8,y8,'r-',markersize=5)
fig.plot(x9,y9,'r-',markersize=5)
fig.plot(x10,y10,'r-',markersize=5)
fig.plot(x11,y11,'r-',markersize=5)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'wo',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'bo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Individual Pings', fontsize=30)

#Show below
#plt.show()

file = 'part3_plot_ln84.svg'
savefig(plt, file)

################################################################################################################################################################################################                

percenterror1,percenterror2 = 0.05, 0.025

k = 0.99

################################################################################################################################################################################################                

plane_hops_5per = []
plane_hops_2per = []

for i in xrange(N):
    plane_hops_5per.append(six_hop_model_wrapped_cauchy(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k,percenterror1,ping_distances,ping_times))
    plane_hops_2per.append(six_hop_model_wrapped_cauchy(last_known_heading,pulauperak[1],pulauperak[0],km_hop,k,percenterror2,ping_distances,ping_times))

################################################################################################################################################################################################                

first_lat_5per = []
two_lat_5per = []
three_lat_5per = []
four_lat_5per = []
five_lat_5per = []
final_lat_5per = []

first_lon_5per = []
two_lon_5per = []
three_lon_5per = []
four_lon_5per = []
five_lon_5per = []
final_lon_5per = []

route1_lat_5per = []
route2_lat_5per = []
route3_lat_5per = []
route4_lat_5per = []

route1_lon_5per = []
route2_lon_5per = []
route3_lon_5per = []
route4_lon_5per = []

for i in xrange(len(plane_hops_5per)):
    first_lat_5per.append(plane_hops_5per[i][0][0])
    first_lon_5per.append(plane_hops_5per[i][1][0])
    two_lat_5per.append(plane_hops_5per[i][0][1])
    two_lon_5per.append(plane_hops_5per[i][1][1])
    three_lat_5per.append(plane_hops_5per[i][0][2])
    three_lon_5per.append(plane_hops_5per[i][1][2])
    four_lat_5per.append(plane_hops_5per[i][0][3])
    four_lon_5per.append(plane_hops_5per[i][1][3])
    five_lat_5per.append(plane_hops_5per[i][0][4])
    five_lon_5per.append(plane_hops_5per[i][1][4])
    final_lat_5per.append(plane_hops_5per[i][0][5])
    final_lon_5per.append(plane_hops_5per[i][1][5])
    
    route1_lat_5per.append(plane_hops_5per[i][0][6])
    route1_lon_5per.append(plane_hops_5per[i][1][6])
    route2_lat_5per.append(plane_hops_5per[i][0][7])
    route2_lon_5per.append(plane_hops_5per[i][1][7])
    route3_lat_5per.append(plane_hops_5per[i][0][8])
    route3_lon_5per.append(plane_hops_5per[i][1][8])
    route4_lat_5per.append(plane_hops_5per[i][0][9])
    route4_lon_5per.append(plane_hops_5per[i][1][9])

################################################################################################################################################################################################                

first_lat_2per = []
two_lat_2per = []
three_lat_2per = []
four_lat_2per = []
five_lat_2per = []
final_lat_2per = []

first_lon_2per = []
two_lon_2per = []
three_lon_2per = []
four_lon_2per = []
five_lon_2per = []
final_lon_2per = []

route1_lat_2per = []
route2_lat_2per = []
route3_lat_2per = []
route4_lat_2per = []

route1_lon_2per = []
route2_lon_2per = []
route3_lon_2per = []
route4_lon_2per = []

for i in xrange(len(plane_hops_2per)):
    first_lat_2per.append(plane_hops_2per[i][0][0])
    first_lon_2per.append(plane_hops_2per[i][1][0])
    two_lat_2per.append(plane_hops_2per[i][0][1])
    two_lon_2per.append(plane_hops_2per[i][1][1])
    three_lat_2per.append(plane_hops_2per[i][0][2])
    three_lon_2per.append(plane_hops_2per[i][1][2])
    four_lat_2per.append(plane_hops_2per[i][0][3])
    four_lon_2per.append(plane_hops_2per[i][1][3])
    five_lat_2per.append(plane_hops_2per[i][0][4])
    five_lon_2per.append(plane_hops_2per[i][1][4])
    final_lat_2per.append(plane_hops_2per[i][0][5])
    final_lon_2per.append(plane_hops_2per[i][1][5])
    
    route1_lat_2per.append(plane_hops_2per[i][0][6])
    route1_lon_2per.append(plane_hops_2per[i][1][6])
    route2_lat_2per.append(plane_hops_2per[i][0][7])
    route2_lon_2per.append(plane_hops_2per[i][1][7])
    route3_lat_2per.append(plane_hops_2per[i][0][8])
    route3_lon_2per.append(plane_hops_2per[i][1][8])
    route4_lat_2per.append(plane_hops_2per[i][0][9])
    route4_lon_2per.append(plane_hops_2per[i][1][9])

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
x5,y5 = fig(circle_lon_211am,circle_lat_211am)
x6,y6 = fig(circle_lon_311am,circle_lat_311am)
x7,y7 = fig(circle_lon_411am,circle_lat_411am)
x8,y8 = fig(circle_lon_511am,circle_lat_511am)
x9,y9 = fig(circle_lon_611am,circle_lat_611am)
x10,y10 = fig(circle_lon_711am,circle_lat_711am)
x11,y11 = fig(circle_lon_811am,circle_lat_811am)

#Add points after each hr
x12,y12 = fig(first_lon_5per,first_lat_5per)
x13,y13 = fig(two_lon_5per,two_lat_5per)
x14,y14 = fig(three_lon_5per,three_lat_5per)
x15,y15 = fig(four_lon_5per,four_lat_5per)
x16,y16 = fig(five_lon_5per,five_lat_5per)
x17,y17 = fig(final_lon_5per,final_lat_5per)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_5per,route1_lat_5per)
x19,y19 = fig(route2_lon_5per,route2_lat_5per)
x20,y20 = fig(route3_lon_5per,route3_lat_5per)
x21,y21 = fig(route4_lon_5per,route4_lat_5per)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw circle showing extent of Inmarsat sat radar detection for each of the pings
fig.plot(x5,y5,'r--',markersize=5,label='1st Ping Arc')
fig.plot(x6,y6,'r-',markersize=5, label='Ping Arcs After Disappearance')
fig.plot(x7,y7,'r-',markersize=5)
fig.plot(x8,y8,'r-',markersize=5)
fig.plot(x9,y9,'r-',markersize=5)
fig.plot(x10,y10,'r-',markersize=5)
fig.plot(x11,y11,'r-',markersize=5)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'wo',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'bo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Individual Pings', fontsize=30)

#Show below
#plt.show()

file = 'part3_plot_ln89.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
x5,y5 = fig(circle_lon_211am,circle_lat_211am)
x6,y6 = fig(circle_lon_311am,circle_lat_311am)
x7,y7 = fig(circle_lon_411am,circle_lat_411am)
x8,y8 = fig(circle_lon_511am,circle_lat_511am)
x9,y9 = fig(circle_lon_611am,circle_lat_611am)
x10,y10 = fig(circle_lon_711am,circle_lat_711am)
x11,y11 = fig(circle_lon_811am,circle_lat_811am)

#Add points after each hr
x12,y12 = fig(first_lon_2per,first_lat_2per)
x13,y13 = fig(two_lon_2per,two_lat_2per)
x14,y14 = fig(three_lon_2per,three_lat_2per)
x15,y15 = fig(four_lon_2per,four_lat_2per)
x16,y16 = fig(five_lon_2per,five_lat_2per)
x17,y17 = fig(final_lon_2per,final_lat_2per)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_2per,route1_lat_2per)
x19,y19 = fig(route2_lon_2per,route2_lat_2per)
x20,y20 = fig(route3_lon_2per,route3_lat_2per)
x21,y21 = fig(route4_lon_2per,route4_lat_2per)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw circle showing extent of Inmarsat sat radar detection for each of the pings
fig.plot(x5,y5,'r--',markersize=5,label='1st Ping Arc')
fig.plot(x6,y6,'r-',markersize=5, label='Ping Arcs After Disappearance')
fig.plot(x7,y7,'r-',markersize=5)
fig.plot(x8,y8,'r-',markersize=5)
fig.plot(x9,y9,'r-',markersize=5)
fig.plot(x10,y10,'r-',markersize=5)
fig.plot(x11,y11,'r-',markersize=5)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'wo',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'bo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Individual Pings', fontsize=30)

#Show below
#plt.show()

file = 'part3_plot_ln93.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)

#Draw coasts
fig.drawcoastlines()

#Draw boundary
fig.drawmapboundary(fill_color='lightblue')

#Fill background
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')

#Draw parallels
parallels = np.arange(lat_min,lat_max,lat_space)
fig.drawparallels(np.arange(lat_min,lat_max,lat_space),labels=[1,1,0,1], fontsize=15)

#Draw meridians
meridians = np.arange(lon_min,lon_max,lon_space)
fig.drawmeridians(np.arange(lon_min,lon_max,lon_space),labels=[1,1,0,1], fontsize=15)

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

#Translate coords into map coord system to plot

#Known 777 Locs
x,y = fig(kualalumpur[1],kualalumpur[0]) #plotted as lon,lat NOT lat,lon -- watch out!!
x2,y2 = fig(igariwaypoint[1],igariwaypoint[0])
x3,y3 = fig(pulauperak[1],pulauperak[0])

#Inmarsat Satellite Loc
x4,y4 = fig(inmarsat[1],inmarsat[0])

#Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
x5,y5 = fig(circle_lon_211am,circle_lat_211am)
x6,y6 = fig(circle_lon_311am,circle_lat_311am)
x7,y7 = fig(circle_lon_411am,circle_lat_411am)
x8,y8 = fig(circle_lon_511am,circle_lat_511am)
x9,y9 = fig(circle_lon_611am,circle_lat_611am)
x10,y10 = fig(circle_lon_711am,circle_lat_711am)
x11,y11 = fig(circle_lon_811am,circle_lat_811am)

#Add points after each hr
x12,y12 = fig(first_lon_2per,first_lat_2per)
x13,y13 = fig(two_lon_2per,two_lat_2per)
x14,y14 = fig(three_lon_2per,three_lat_2per)
x15,y15 = fig(four_lon_2per,four_lat_2per)
x16,y16 = fig(five_lon_2per,five_lat_2per)
x17,y17 = fig(final_lon_2per,final_lat_2per)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_2per,route1_lat_2per)
x19,y19 = fig(route2_lon_2per,route2_lat_2per)
x20,y20 = fig(route3_lon_2per,route3_lat_2per)
x21,y21 = fig(route4_lon_2per,route4_lat_2per)

# plot coords w/ filled circles
fig.plot(x,y,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x2,y2,'bo',markersize=10)
fig.plot(x3,y3,'go',markersize=10,label='MH370 Last Known Coords')
fig.plot(x4,y4,'ro',markersize=10,label='Inmarsat 3-F1')

#Draw circle showing extent of Inmarsat sat radar detection for each of the pings
fig.plot(x5,y5,'r--',markersize=5,label='1st Ping Arc')
fig.plot(x6,y6,'r-',markersize=5, label='Ping Arcs After Disappearance')
fig.plot(x7,y7,'r-',markersize=5)
fig.plot(x8,y8,'r-',markersize=5)
fig.plot(x9,y9,'r-',markersize=5)
fig.plot(x10,y10,'r-',markersize=5)
fig.plot(x11,y11,'r-',markersize=5)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'wo',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'bo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Draw arrows showing flight path
arrow1 = plt.arrow(x,y,x2-x,y2-y,linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2,y2,x3-x2,y3-y2,linewidth=3,color='blue',linestyle='dashed',label='flight path')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Great Circle Path on Most Stringent Scenario to Constrain Moving West', fontsize=15)

#Show below
#plt.show()

file = 'part3_plot_ln92.svg'
savefig(plt, file)

################################################################################################################################################################################################                
