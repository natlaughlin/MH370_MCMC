description=u'''
Created on Apr 15, 2014

Nat Laughlin
http://natlaughlin.com
http://github.com/natlaughlin

Based on:

http://nbviewer.ipython.org/github/myhrvold/MH370_MCMC/blob/master/MH370_MC_ConorMyhrvold-V3.ipynb?create=1

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

parser = optparse.OptionParser(formatter=PlainHelpFormatter(),description=description)

default = 255.136
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

default = os.getcwd()
help = """output directory (default: {0})
""".format(default)
parser.add_option('-o', '--output-directory', dest='output_directory', default=default, help=help)

(options, args) = parser.parse_args()

if not os.path.exists(options.output_directory):
    os.makedirs(options.output_directory)
    
parsed_options = ", ".join(['{0}: {1}'.format(key, value) for (key, value) in options.__dict__.items()])

print parsed_options

def savefig(plt, file):
    fn = os.path.join(options.output_directory, file)
    plt.figtext(0,1,parsed_options, fontsize=10, horizontalalignment='left', verticalalignment='top')
    plt.savefig(fn)
    plt.close()

################################################################################################################################################################################################                
# PART 3
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

#Here's the ping arc distance function
"""
Computes the ping arc distance from the satellite to the plane.
Returns the angle in degrees.
"""
def satellite_calc(radius,orbit,angle):
    interim = (np.sqrt(-radius**2+orbit**2*(1./np.cos(angle)**2))-orbit*np.tan(angle))/np.float(orbit+radius)
    return np.degrees(2*np.arctan(interim))


# Here's the ellipse function (which is more realistic and favored over the simple circle function): 
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


#Here's our Haversine equation formula
"""
Haversine equation.
Computes the great circle distance between two pairs of longitude/latitude.
Returns the distance in m or km depending on input (I use meters.) 
"""
def haversine(r,lat1,lon1,lat2,lon2):
    d = 2.0*r*np.arcsin(np.sqrt(np.sin(np.radians(lat2-lat1)/2.0)**2+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(np.radians(lon2-lon1)/2.0)**2))
    return d   


# Convert the Inmarsat line location and error bounds into probabilities
"""
we center a normal probability distribution upon the location of the radius line.
d = a distance
r = a radius
lat1, lon1 is a latitude and longitude
lat2, lon2 we iterate through for our function
err is the error
"""
def ping_prob_normal(lat1,lon1,lat2,lon2,err,d,r): 
    return np.exp(-0.5*((haversine(r,lat1,lon1,lat2,lon2)-d)/(err*d))**2)/(d*np.sqrt(2*np.pi)) 


# Here's the function that picks a new location for the plane at each time step:
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


# Here's the same-heading vectors from the last ping, to whereever the plane may finally be located.
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


#MCMC Functions

#Best of Version 1
# Here's the main Monte Carlo function which puts it all together:
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


#Best of Version 2
# Here's the main Monte Carlo function which puts it all together:
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
    
    new_plane_lat[6] = route1[0] 
    new_plane_lat[7] = route2[0] 
    new_plane_lat[8] = route3[0] 
    new_plane_lat[9] = route4[0] 
    new_plane_lon[6] = route1[1]
    new_plane_lon[7] = route2[1] 
    new_plane_lon[8] = route3[1] 
    new_plane_lon[9] = route4[1]
    
    return new_plane_lat,new_plane_lon

################################################################################################################################################################################################                

eq_deg_km = 111.32 # number of km/degree at eq Source: http://en.wikipedia.org/wiki/Decimal_degrees
earth_radius = 6371 #in km, http://en.wikipedia.org/wiki/Earth

#Inmarsat satellite information
sat_height = 42170 #Inmarsat satellite height off ground, in meters
elevation_angle = np.radians(40) #elevation angle of satellite; convert degrees to radians. Source: NYT Hong Kong Bureau

################################################################################################################################################################################################                

# The Inmarsat satellite is at 0,64.5 -- it's geostationary.
inmarsat = [0, 64.5]

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

#ping arc distances as calculated in Mathematica
ping_distances = np.array([4036.99, 4194.65, 4352.32, 4509.99, 4667.65, 4825.32])
ping_times = np.array([0.9333, 1, 1, 1, 1, 1]) #make 1st hop slightly smaller owing to time difference
ping_arcs = np.array([34.8485, 36.2649, 37.6812, 39.0976, 40.5139, 41.9303, 43.3466])

################################################################################################################################################################################################                

ping_arc_dist = satellite_calc(earth_radius,sat_height,elevation_angle)
print "ping arc distance in degrees:", ping_arc_dist
dist_from_sat = earth_radius*np.radians(satellite_calc(earth_radius,sat_height,elevation_angle))
print "distance from satellite", dist_from_sat

################################################################################################################################################################################################                

#create lat/long grid of distance
lat_min = -50 #50 S
lat_max = 50  #50 N
lon_min = 50  #50 E
lon_max = 140 #130 E
lat_space = 5 #spacing for plotting latitudes and longitudes
lon_space = 5

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

"""
To conserve space I will write the first part of the plot as a function that I'll run, since these will remain the same.
I could pass in parameters I'd want to change, as variables as well, but it isn't necessary for my purposes.
"""
def figure_function_all_pings(fig):    
        
    #Draw coasts
    fig.drawcoastlines()
    
    #Draw boundary
    fig.drawmapboundary(fill_color='lightblue')
    
    #Fill background
    fig.fillcontinents(color='white',lake_color='lightblue')
    
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

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)
    
#Call figure function
figure_function_all_pings(fig)

#Put other plots in here -- they don't have to be in any strict order necessarily.
# (We don't have any right now)

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Individual Pings', fontsize=30)

#Show below
#plt.show()
file = 'plot_ln52.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#plot the 5th ping arc

circle_pts = make_circle(ping_arc_dist,360,64.5,0)

circle_lat = []
for i in xrange(len(circle_pts)):
    circle_lat.append(circle_pts[i][0])

circle_lon = []
for i in xrange(len(circle_pts)):
    circle_lon.append(circle_pts[i][1])

################################################################################################################################################################################################                

# 2.5% error

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

# 5% error

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

"""
Version 1 figure creation -- equivalent to the above. I show the 2.5% & 5% errors.
"""
def figure_function_fifth_ping(fig):    
        
    #Draw coasts
    fig.drawcoastlines()
    
    #Draw boundary
    fig.drawmapboundary(fill_color='lightblue')
    
    #Fill background
    fig.fillcontinents(color='white',lake_color='lightblue')
    
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
    
    #Add circle coords -- these are for each ping. will not plot the 2.5 and 5% error
    x5,y5 = fig(circle_lon,circle_lat)
    x6,y6 = fig(circle_lon_err1_2per,circle_lat_err1_2per)
    x7,y7 = fig(circle_lon_err2_2per,circle_lat_err2_2per)
    x8,y8 = fig(circle_lon_err1_5per,circle_lat_err1_5per)
    x9,y9 = fig(circle_lon_err2_5per,circle_lat_err2_5per)
    
    #Draw circle showing extent of Inmarsat sat radar detection for each of the pings
    fig.plot(x5,y5,'r-',markersize=5,label='Last Ping Arc')
    fig.plot(x6,y6,'r--',markersize=5, label='2.5% and 5% Error')
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

################################################################################################################################################################################################                



#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)
    
#Call figure function
figure_function_fifth_ping(fig)

#Put other plots in here -- they don't have to be in any strict order necessarily.
# (We don't have any right now)

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('Inmarsat Ping Estimation -- Last Ping', fontsize=30) #at time, 5th thought to be last ping! 
                                                               #note: it's still in the *right location*
                                                               #basically it assumes a slower speed

#Show below
#plt.show()
file = 'plot_ln212.svg'
savefig(plt, file)

################################################################################################################################################################################################                

last_known_heading = 255.136 #calculated in Mathematica from MH370's two last publically known locations:
                             #when it deviated from its flight path, and when it was last detected by Malyasian military radar
                             #0 degrees is due north, so this is basically to the west (270 degrees), but slightly south

km_hop = 905 #assuming 1 hr intervals, at 905 km/hr which is 777 cruising speed -- use for test case
             # max speed of a Boeing 777 is 950 km/hr FYI

N = 10000 #define number of simulations to run

percenterror1,percenterror2 = 0.05, 0.025 #ping arc standard deviations -- 2.5% and 5% of last ping arc

std_dev = 30 # allows for some turning each time

################################################################################################################################################################################################                

#override simulation params with cli options

if options.km_hop:
    km_hop = float(options.km_hop)
    
if options.last_known_heading:
    last_known_heading = float(options.last_known_heading)

if options.simulations:
    N = int(options.simulations)
    
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
    
#Call figure function
figure_function_all_pings(fig)

#Put other plots in here -- they don't have to be in any strict order necessarily.

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

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

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'go',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'mo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('5% Ping Arc Error', fontsize=20)

#Show below
#plt.show()
file = 'plot_ln168.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)
    
#Call figure function
figure_function_all_pings(fig)

#Put other plots in here -- they don't have to be in any strict order necessarily.

#Draw great circle to show path autopilot would have taken
fig.drawgreatcircle(pulauperak[1],pulauperak[0],85,-40,linewidth=3,color='black',label='Great Circle Path')

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

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1 hrs')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2 hrs')
fig.plot(x14,y14,'go',markersize=5,label='after 3 hrs')
fig.plot(x15,y15,'mo',markersize=5,label='after 4 hrs')
fig.plot(x16,y16,'go',markersize=5,label='after 5 hrs')
fig.plot(x17,y17,'ro',markersize=7,label='after 6 hrs')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('2.5% Ping Arc Error', fontsize=20)

#Show below
#plt.show()
file = 'plot_ln169.svg'
savefig(plt, file)

################################################################################################################################################################################################                

plane_hops_5per_v1 = []
plane_hops_2per_v1 = []

for i in xrange(N):
    plane_hops_5per_v1.append(five_hop_model_normal(last_known_heading,pulauperak[1],pulauperak[0],km_hop,std_dev,percenterror1))
    plane_hops_2per_v1.append(five_hop_model_normal(last_known_heading,pulauperak[1],pulauperak[0],km_hop,std_dev,percenterror2))

################################################################################################################################################################################################                

first_lat_5per_v1 = []
two_lat_5per_v1 = []
three_lat_5per_v1 = []
four_lat_5per_v1 = []
final_lat_5per_v1 = []

first_lon_5per_v1 = []
two_lon_5per_v1 = []
three_lon_5per_v1 = []
four_lon_5per_v1 = []
final_lon_5per_v1 = []

route1_lat_5per_v1 = []
route2_lat_5per_v1 = []
route3_lat_5per_v1 = []
route4_lat_5per_v1 = []

route1_lon_5per_v1 = []
route2_lon_5per_v1 = []
route3_lon_5per_v1 = []
route4_lon_5per_v1 = []

for i in xrange(len(plane_hops_5per_v1)):
    first_lat_5per_v1.append(plane_hops_5per_v1[i][0][0])
    first_lon_5per_v1.append(plane_hops_5per_v1[i][1][0])
    two_lat_5per_v1.append(plane_hops_5per_v1[i][0][1])
    two_lon_5per_v1.append(plane_hops_5per_v1[i][1][1])
    three_lat_5per_v1.append(plane_hops_5per_v1[i][0][2])
    three_lon_5per_v1.append(plane_hops_5per_v1[i][1][2])
    four_lat_5per_v1.append(plane_hops_5per_v1[i][0][3])
    four_lon_5per_v1.append(plane_hops_5per_v1[i][1][3])
    final_lat_5per_v1.append(plane_hops_5per_v1[i][0][4])
    final_lon_5per_v1.append(plane_hops_5per_v1[i][1][4])
    
    route1_lat_5per_v1.append(plane_hops_5per_v1[i][0][5])
    route1_lon_5per_v1.append(plane_hops_5per_v1[i][1][5])
    route2_lat_5per_v1.append(plane_hops_5per_v1[i][0][6])
    route2_lon_5per_v1.append(plane_hops_5per_v1[i][1][6])
    route3_lat_5per_v1.append(plane_hops_5per_v1[i][0][7])
    route3_lon_5per_v1.append(plane_hops_5per_v1[i][1][7])
    route4_lat_5per_v1.append(plane_hops_5per_v1[i][0][8])
    route4_lon_5per_v1.append(plane_hops_5per_v1[i][1][8])

################################################################################################################################################################################################                

first_lat_2per_v1 = []
two_lat_2per_v1 = []
three_lat_2per_v1 = []
four_lat_2per_v1 = []
final_lat_2per_v1 = []

first_lon_2per_v1 = []
two_lon_2per_v1 = []
three_lon_2per_v1 = []
four_lon_2per_v1 = []
final_lon_2per_v1 = []

route1_lat_2per_v1 = []
route2_lat_2per_v1 = []
route3_lat_2per_v1 = []
route4_lat_2per_v1 = []

route1_lon_2per_v1 = []
route2_lon_2per_v1 = []
route3_lon_2per_v1 = []
route4_lon_2per_v1 = []

for i in xrange(len(plane_hops_2per_v1)):
    first_lat_2per_v1.append(plane_hops_2per_v1[i][0][0])
    first_lon_2per_v1.append(plane_hops_2per_v1[i][1][0])
    two_lat_2per_v1.append(plane_hops_2per_v1[i][0][1])
    two_lon_2per_v1.append(plane_hops_2per_v1[i][1][1])
    three_lat_2per_v1.append(plane_hops_2per_v1[i][0][2])
    three_lon_2per_v1.append(plane_hops_2per_v1[i][1][2])
    four_lat_2per_v1.append(plane_hops_2per_v1[i][0][3])
    four_lon_2per_v1.append(plane_hops_2per_v1[i][1][3])
    final_lat_2per_v1.append(plane_hops_2per_v1[i][0][4])
    final_lon_2per_v1.append(plane_hops_2per_v1[i][1][4])
    
    route1_lat_2per_v1.append(plane_hops_2per_v1[i][0][5])
    route1_lon_2per_v1.append(plane_hops_2per_v1[i][1][5])
    route2_lat_2per_v1.append(plane_hops_2per_v1[i][0][6])
    route2_lon_2per_v1.append(plane_hops_2per_v1[i][1][6])
    route3_lat_2per_v1.append(plane_hops_2per_v1[i][0][7])
    route3_lon_2per_v1.append(plane_hops_2per_v1[i][1][7])
    route4_lat_2per_v1.append(plane_hops_2per_v1[i][0][8])
    route4_lon_2per_v1.append(plane_hops_2per_v1[i][1][8])

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)
    
#Call figure function
figure_function_fifth_ping(fig)

#Put other plots in here -- they don't have to be in any strict order necessarily.

#Add points after each hr
x12,y12 = fig(first_lon_5per_v1,first_lat_5per_v1)
x13,y13 = fig(two_lon_5per_v1,two_lat_5per_v1)
x14,y14 = fig(three_lon_5per_v1,three_lat_5per_v1)
x15,y15 = fig(four_lon_5per_v1,four_lat_5per_v1)
x17,y17 = fig(final_lon_5per_v1,final_lat_5per_v1)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_5per_v1,route1_lat_5per_v1)
x19,y19 = fig(route2_lon_5per_v1,route2_lat_5per_v1)
x20,y20 = fig(route3_lon_5per_v1,route3_lat_5per_v1)
x21,y21 = fig(route4_lon_5per_v1,route4_lat_5per_v1)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1st ping')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2nd ping')
fig.plot(x14,y14,'go',markersize=5,label='after 3rd ping')
fig.plot(x15,y15,'mo',markersize=5,label='after 4th ping')
fig.plot(x17,y17,'ro',markersize=7,label='after final ping')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('5% Ping Arc Error', fontsize=20)

#Show below
#plt.show()
file = 'plot_ln213.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)
    
#Call figure function
figure_function_fifth_ping(fig)

#Put other plots in here -- they don't have to be in any strict order necessarily.

#Add points after each hr
x12,y12 = fig(first_lon_2per_v1,first_lat_2per_v1)
x13,y13 = fig(two_lon_2per_v1,two_lat_2per_v1)
x14,y14 = fig(three_lon_2per_v1,three_lat_2per_v1)
x15,y15 = fig(four_lon_2per_v1,four_lat_2per_v1)
x17,y17 = fig(final_lon_2per_v1,final_lat_2per_v1)

#Add ultimate locations of MH370
x18,y18 = fig(route1_lon_2per_v1,route1_lat_2per_v1)
x19,y19 = fig(route2_lon_2per_v1,route2_lat_2per_v1)
x20,y20 = fig(route3_lon_2per_v1,route3_lat_2per_v1)
x21,y21 = fig(route4_lon_2per_v1,route4_lat_2per_v1)

#Add monte carlo points
fig.plot(x12,y12,'yo',markersize=5,label='after 1st ping')
fig.plot(x13,y13,'mo',markersize=5,label= 'after 2nd ping')
fig.plot(x14,y14,'go',markersize=5,label='after 3rd ping')
fig.plot(x15,y15,'mo',markersize=5,label='after 4th ping')
fig.plot(x17,y17,'ro',markersize=7,label='after final ping')

#Plot ultimate locations of MH370
fig.plot(x18,y18,'bo',markersize=5,label='in final hr')
fig.plot(x19,y19,'bo',markersize=5)
fig.plot(x20,y20,'bo',markersize=5)
fig.plot(x21,y21,'bo',markersize=5)

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('2.5% Ping Arc Error', fontsize=20)

#Show below
#plt.show()
file = 'plot_ln214.svg'
savefig(plt, file)

################################################################################################################################################################################################                

# instead of using mode, look at histogram of values into bins which will effectively 
# be like rounding all of the values and then taking the mode of the rounded values

# 2.5%, V2
fig = plt.figure(figsize=[10,6])
plt.hist(final_lat_2per, bins=50)
plt.title('Last Ping Locations of MH370, 2.5% error, Version 2',fontsize=13)
plt.ylabel('# of instances',fontsize=12)
plt.xlabel('Latitude',fontsize=12)
plt.xlim([lat_min,lat_max])
#plt.show()
file = 'plot_ln89_1.svg'
savefig(plt, file)

# 5%, V2
fig = plt.figure(figsize=[10,6])
plt.hist(final_lat_5per, bins=50, color='red')
plt.title('Last Ping Locations of MH370, 5% error, Version 2',fontsize=13)
plt.ylabel('# of instances',fontsize=12)
plt.xlabel('Latitude',fontsize=12)
plt.xlim([lat_min,lat_max])
#plt.show()
file = 'plot_ln89_2.svg'
savefig(plt, file)

# 2.5%, V2
fig = plt.figure(figsize=[10,6])
plt.hist(final_lon_2per, bins=50)
plt.title('Last Ping Locations of MH370, 2.5% error, Version 2',fontsize=13)
plt.ylabel('# of instances',fontsize=12)
plt.xlabel('Longitude',fontsize=12)
plt.xlim([lon_min,lon_max])
#plt.show()
file = 'plot_ln89_3.svg'
savefig(plt, file)

# 5%, V2
fig = plt.figure(figsize=[10,6])
plt.hist(final_lon_5per, bins=50, color='red')
plt.title('Last Ping Locations of MH370, 5% error, Version 2',fontsize=13)
plt.ylabel('# of instances',fontsize=12)
plt.xlabel('Longitude',fontsize=12)
plt.xlim([lon_min,lon_max])
#plt.show()
file = 'plot_ln89_4.svg'
savefig(plt, file)

################################################################################################################################################################################################                

indices_filter_2per = []
indices_filter_5per = []

for i in xrange(len(final_lat_2per)):
    if final_lat_2per[i] < 0: #if southern track
        indices_filter_2per.append(i)

for i in xrange(len(final_lon_2per)):
    if final_lon_2per[i] > 80 and final_lon_2per[i] < 95: #if not way too west or east
        indices_filter_2per.append(i)
        
for i in xrange(len(final_lat_5per)):
    if final_lat_5per[i] < 0:
        indices_filter_5per.append(i)

for i in xrange(len(final_lon_5per)):
    if final_lon_5per[i] > 80 and final_lon_5per[i] < 100:
        indices_filter_5per.append(i)
        
#now filter out unique values in case of lat/lon overlap, i.e. they're both out of bounds
print "before filter:"
print "2.5% err:", len(indices_filter_2per)
print "5% err:", len(indices_filter_5per)
#print indices_to_filter_2per
indices_filter_2per = np.unique(indices_filter_2per)
indices_filter_5per = np.unique(indices_filter_5per)
#convert back from np array to list
indices_filter_2per = list(indices_filter_2per)
indices_filter_5per = list(indices_filter_5per)
#print out number of unique simulations to keep (we've stored their exact indices)
print "after filter:"
print "2.5% err:", len(indices_filter_2per)
print "5% err:", len(indices_filter_5per)

################################################################################################################################################################################################                

filter_final_lat_2per = [final_lat_2per[i] for i in indices_filter_2per] 
filter_final_lon_2per = [final_lon_2per[i] for i in indices_filter_2per] 

filter_final_lat_5per = [final_lat_5per[i] for i in indices_filter_5per] 
filter_final_lon_5per = [final_lon_5per[i] for i in indices_filter_5per]

################################################################################################################################################################################################                

print "2.5% err, lat values filtered: ", len(final_lat_2per)-len(filter_final_lat_2per)
print "2.5% err, lon values filtered: ", len(final_lon_2per)-len(filter_final_lon_2per)
print "5% err, lat values filtered: ", len(final_lat_5per)-len(filter_final_lat_5per)
print "5% err, lon values filtered: ", len(final_lon_5per)-len(filter_final_lon_5per)

################################################################################################################################################################################################                

mean_lat_2per = np.mean(filter_final_lat_2per)
mean_lon_2per = np.mean(filter_final_lon_2per)

median_lat_2per = np.median(filter_final_lat_2per) 
median_lon_2per = np.median(filter_final_lon_2per)

variance_lat_2per = np.var(filter_final_lat_2per)
variance_lon_2per = np.var(filter_final_lon_2per)

print "Version 2, 2.5% err, mean latitude is:", mean_lat_2per
print "Version 2, 2.5% err, mean longitude is:", mean_lon_2per
print "Version 2, 2.5% err, median latitude is:", median_lat_2per
print "Version 2, 2.5% err, median longitude is:", median_lon_2per

print "--------------------"

print "Version 2, 2.5% err, lat variance:", variance_lat_2per
print "Version 2, 2.5% err, lon variance:", variance_lon_2per

################################################################################################################################################################################################                

mean_lat_5per = np.mean(filter_final_lat_5per)
mean_lon_5per = np.mean(filter_final_lon_5per)

median_lat_5per = np.median(filter_final_lat_5per) 
median_lon_5per = np.median(filter_final_lon_5per)

variance_lat_5per = np.var(filter_final_lat_5per)
variance_lon_5per = np.var(filter_final_lon_5per)

print "Version 2, 5% err, mean latitude is:", mean_lat_5per
print "Version 2, 5% err, mean longitude is:", mean_lon_5per
print "Version 2, 5% err, median latitude is:", median_lat_5per
print "Version 2, 5% err, median longitude is:", median_lon_5per

print "--------------------"

print "Version 2, 5% err, lat variance:", variance_lat_5per
print "Version 2, 5% err, lon variance:", variance_lon_5per

################################################################################################################################################################################################                

# 2.5%, V1
fig = plt.figure(figsize=[10,6])
plt.hist(final_lat_2per_v1, bins=50)
plt.title('Last Ping Locations of MH370, 2.5% error, Version 1',fontsize=13)
plt.ylabel('# of instances',fontsize=12)
plt.xlabel('Latitude',fontsize=12)
plt.xlim([lat_min,lat_max])
#plt.show()
file = 'plot_ln152_1.svg'
savefig(plt, file)

# 5%, V1
fig = plt.figure(figsize=[10,6])
plt.hist(final_lat_5per_v1, bins=50, color='red')
plt.title('Last Ping Locations of MH370, 5% error, Version 1',fontsize=13)
plt.ylabel('# of instances',fontsize=12)
plt.xlabel('Latitude',fontsize=12)
plt.xlim([lat_min,lat_max])
#plt.show()
file = 'plot_ln152_2.svg'
savefig(plt, file)

# 2.5%, V1
fig = plt.figure(figsize=[10,6])
plt.hist(final_lon_2per_v1, bins=50)
plt.title('Last Ping Locations of MH370, 2.5% error, Version 1',fontsize=13)
plt.ylabel('# of instances',fontsize=12)
plt.xlabel('Longitude',fontsize=12)
plt.xlim([lon_min,lon_max])
#plt.show()
file = 'plot_ln152_3.svg'
savefig(plt, file)

# 5%, V1
fig = plt.figure(figsize=[10,6])
plt.hist(final_lon_5per_v1, bins=50, color='red')
plt.title('Last Ping Locations of MH370, 5% error, Version 1',fontsize=13)
plt.ylabel('# of instances',fontsize=12)
plt.xlabel('Longitude',fontsize=12)
plt.xlim([lon_min,lon_max])
#plt.show()
file = 'plot_ln152_4.svg'
savefig(plt, file)

################################################################################################################################################################################################                

indices_filter_2per_v1 = []
indices_filter_5per_v1 = []

for i in xrange(len(final_lat_2per_v1)):
    if final_lat_2per_v1[i] < -20 and final_lat_2per_v1[i] > -38: #if southern track
        indices_filter_2per_v1.append(i)

for i in xrange(len(final_lon_2per_v1)):
    if final_lon_2per_v1[i] > 95 and final_lon_2per_v1[i] < 105: #if not way too west or east
        indices_filter_2per_v1.append(i)
        
for i in xrange(len(final_lat_5per_v1)):
    if final_lat_5per_v1[i] < -20 and final_lat_5per_v1[i] > -38: #if southern track
        indices_filter_5per_v1.append(i)

for i in xrange(len(final_lon_5per_v1)):
    if final_lon_5per_v1[i] > 90 and final_lon_5per_v1[i] < 110:
        indices_filter_5per_v1.append(i)
        
#now filter out unique values in case of lat/lon overlap, i.e. they're both out of bounds
print "before filter:"
print "2.5% err:", len(indices_filter_2per_v1)
print "5% err:", len(indices_filter_5per_v1)
#print indices_to_filter_2per
indices_filter_2per_v1 = np.unique(indices_filter_2per_v1)
indices_filter_5per_v1 = np.unique(indices_filter_5per_v1)
#convert back from np array to list
indices_filter_2per_v1 = list(indices_filter_2per_v1)
indices_filter_5per_v1 = list(indices_filter_5per_v1)
#print out number of unique simulations to keep (we've stored their exact indices)
print "after filter:"
print "2.5% err:", len(indices_filter_2per_v1)
print "5% err:", len(indices_filter_5per_v1)

################################################################################################################################################################################################                

filter_final_lat_2per_v1 = [final_lat_2per_v1[i] for i in indices_filter_2per_v1] 
filter_final_lon_2per_v1 = [final_lon_2per_v1[i] for i in indices_filter_2per_v1] 

filter_final_lat_5per_v1 = [final_lat_5per_v1[i] for i in indices_filter_5per_v1] 
filter_final_lon_5per_v1 = [final_lon_5per_v1[i] for i in indices_filter_5per_v1]

################################################################################################################################################################################################                

print "2.5% err, lat values filtered: ", len(final_lat_2per_v1)-len(filter_final_lat_2per_v1)
print "2.5% err, lon values filtered: ", len(final_lon_2per_v1)-len(filter_final_lon_2per_v1)
print "5% err, lat values filtered: ", len(final_lat_5per_v1)-len(filter_final_lat_5per_v1)
print "5% err, lon values filtered: ", len(final_lon_5per_v1)-len(filter_final_lon_5per_v1)


################################################################################################################################################################################################                

mean_lat_2per_v1 = np.mean(filter_final_lat_2per_v1)
mean_lon_2per_v1 = np.mean(filter_final_lon_2per_v1)

median_lat_2per_v1 = np.median(filter_final_lat_2per_v1) 
median_lon_2per_v1 = np.median(filter_final_lon_2per_v1)

variance_lat_2per_v1 = np.var(filter_final_lat_2per_v1)
variance_lon_2per_v1 = np.var(filter_final_lon_2per_v1)

print "Version 1, 2.5% err, mean latitude is:", mean_lat_2per_v1
print "Version 1, 2.5% err, mean longitude is:", mean_lon_2per_v1
print "Version 1, 2.5% err, median latitude is:", median_lat_2per_v1
print "Version 1, 2.5% err, median longitude is:", median_lon_2per_v1

print "--------------------"

print "Version 1, 2.5% err, lat variance:", variance_lat_2per_v1
print "Version 1, 2.5% err, lon variance:", variance_lon_2per_v1

################################################################################################################################################################################################                

mean_lat_5per_v1 = np.mean(filter_final_lat_5per_v1)
mean_lon_5per_v1 = np.mean(filter_final_lon_5per_v1)

median_lat_5per_v1 = np.median(filter_final_lat_5per_v1) 
median_lon_5per_v1 = np.median(filter_final_lon_5per_v1)

variance_lat_5per_v1 = np.var(filter_final_lat_5per_v1)
variance_lon_5per_v1 = np.var(filter_final_lon_5per_v1)

print "Version 1, 5% err, mean latitude is:", mean_lat_5per_v1
print "Version 1, 5% err, mean longitude is:", mean_lon_5per_v1
print "Version 1, 5% err, median latitude is:", median_lat_5per_v1
print "Version 1, 5% err, median longitude is:", median_lon_5per_v1

print "--------------------"

print "Version 1, 5% err, lat variance:", variance_lat_5per_v1
print "Version 1, 5% err, lon variance:", variance_lon_5per_v1

################################################################################################################################################################################################                

#make points for circles 

#Version 2 model
hr_circle_2per_mean = make_circle(km_hop/eq_deg_km,360,mean_lon_2per,mean_lat_2per)
hr_circle_5per_mean = make_circle(km_hop/eq_deg_km,360,mean_lon_5per,mean_lat_5per)
hr_circle_2per_median = make_circle(km_hop/eq_deg_km,360,median_lon_2per,median_lat_2per)
hr_circle_5per_median = make_circle(km_hop/eq_deg_km,360,median_lon_5per,median_lat_5per)

#Version 1 model
hr_circle_2per_mean_v1 = make_circle(km_hop/eq_deg_km,360,mean_lon_2per_v1,mean_lat_2per_v1)
hr_circle_5per_mean_v1 = make_circle(km_hop/eq_deg_km,360,mean_lon_5per_v1,mean_lat_5per_v1)
hr_circle_2per_median_v1 = make_circle(km_hop/eq_deg_km,360,median_lon_2per_v1,median_lat_2per_v1)
hr_circle_5per_median_v1 = make_circle(km_hop/eq_deg_km,360,median_lon_5per_v1,median_lat_5per_v1)

#initialize lat & lon lists

#Version 2 model
circle_lon_2per_mean = []
circle_lat_2per_mean = []
circle_lat_5per_mean = []
circle_lon_5per_mean = []
circle_lat_2per_median = []
circle_lon_2per_median = []
circle_lat_5per_median = []
circle_lon_5per_median = []

#Version 1 model
circle_lon_2per_mean_v1 = []
circle_lat_2per_mean_v1 = []
circle_lat_5per_mean_v1 = []
circle_lon_5per_mean_v1 = []
circle_lat_2per_median_v1 = []
circle_lon_2per_median_v1 = []
circle_lat_5per_median_v1 = []
circle_lon_5per_median_v1 = []

for i in xrange(len(hr_circle_2per_mean)): #they're all the same length so just do it once
    circle_lon_2per_mean.append(hr_circle_2per_mean[i][1])
    circle_lat_2per_mean.append(hr_circle_2per_mean[i][0])
    
    circle_lon_2per_median.append(hr_circle_2per_median[i][1])
    circle_lat_2per_median.append(hr_circle_2per_median[i][0])

    circle_lon_5per_mean.append(hr_circle_5per_mean[i][1])
    circle_lat_5per_mean.append(hr_circle_5per_mean[i][0])
    
    circle_lon_5per_median.append(hr_circle_5per_median[i][1])
    circle_lat_5per_median.append(hr_circle_5per_median[i][0])
    
    
    circle_lon_2per_mean_v1.append(hr_circle_2per_mean_v1[i][1])
    circle_lat_2per_mean_v1.append(hr_circle_2per_mean_v1[i][0])
    
    circle_lon_2per_median_v1.append(hr_circle_2per_median_v1[i][1])
    circle_lat_2per_median_v1.append(hr_circle_2per_median_v1[i][0])

    circle_lon_5per_mean_v1.append(hr_circle_5per_mean_v1[i][1])
    circle_lat_5per_mean_v1.append(hr_circle_5per_mean_v1[i][0])
    
    circle_lon_5per_median_v1.append(hr_circle_5per_median_v1[i][1])
    circle_lat_5per_median_v1.append(hr_circle_5per_median_v1[i][0])

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)
    
#Call figure function
figure_function_all_pings(fig)

#Put other plots in here -- they don't have to be in any strict order necessarily.
# (We don't have any right now)

#Plot average coordinates
x12,y12 = fig(mean_lon_5per, mean_lat_5per)
x13,y13 = fig(median_lon_5per, median_lat_5per)
x14,y14 = fig(mean_lon_2per, mean_lat_2per)
x15,y15 = fig(median_lon_2per, median_lat_2per)

#Plot 1 hr radius circles for average coordinates
x16,y16 = fig(circle_lon_5per_mean,circle_lat_5per_mean)
x17,y17 = fig(circle_lon_5per_median,circle_lat_5per_median)
x18,y18 = fig(circle_lon_2per_mean,circle_lat_2per_mean)
x19,y19 = fig(circle_lon_2per_median,circle_lat_2per_median)

#Draw points showing mean & median locations
fig.plot(x12,y12,'ko',markersize=9,label='5% err, mean')
fig.plot(x13,y13,'yo',markersize=9,label='5% err, median')
fig.plot(x14,y14,'mo',markersize=9,label='2.5% err, mean')
fig.plot(x15,y15,'co',markersize=9,label='2.5% err, median')

#Draw areas where plane is most likely to be -- (up to) the last hour  of flight
fig.plot(x16,y16,'k--',markersize=7, label='5% err, mean')
fig.plot(x17,y17,'y--',markersize=7, label='5% err, median')
fig.plot(x18,y18,'m--',markersize=7, label='2.5% err, mean')
fig.plot(x19,y19,'c--',markersize=7, label='2.5% err, median')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Final Locations -- Where to Search', fontsize=30)

#Show below
#plt.show()
file = 'plot_ln208.svg'
savefig(plt, file)

################################################################################################################################################################################################                

#Set figure size
fig = plt.figure(figsize=[30,20])

#Setup Basemap
fig = Basemap(width=10000000,height=18000000,projection='lcc',resolution='c',lat_0=10,lon_0=90,suppress_ticks=True)
    
#Call figure function
figure_function_fifth_ping(fig)

#Put other plots in here -- they don't have to be in any strict order necessarily.
# (We don't have any right now)

#Plot average coordinates
x12,y12 = fig(mean_lon_5per_v1, mean_lat_5per_v1)
x13,y13 = fig(median_lon_5per_v1, median_lat_5per_v1)
x14,y14 = fig(mean_lon_2per_v1, mean_lat_2per_v1)
x15,y15 = fig(median_lon_2per_v1, median_lat_2per_v1)

#Plot 1 hr radius circles for average coordinates
x16,y16 = fig(circle_lon_5per_mean_v1,circle_lat_5per_mean_v1)
x17,y17 = fig(circle_lon_5per_median_v1,circle_lat_5per_median_v1)
x18,y18 = fig(circle_lon_2per_mean_v1,circle_lat_2per_mean_v1)
x19,y19 = fig(circle_lon_2per_median_v1,circle_lat_2per_median_v1)

#Draw points showing mean & median locations
fig.plot(x12,y12,'ko',markersize=9,label='5% err, mean')
fig.plot(x13,y13,'yo',markersize=9,label='5% err, median')
fig.plot(x14,y14,'mo',markersize=9,label='2.5% err, mean')
fig.plot(x15,y15,'co',markersize=9,label='2.5% err, median')

#Draw areas where plane is most likely to be -- (up to) the last hour  of flight
fig.plot(x16,y16,'k--',markersize=7, label='5% err, mean')
fig.plot(x17,y17,'y--',markersize=7, label='5% err, median')
fig.plot(x18,y18,'m--',markersize=7, label='2.5% err, mean')
fig.plot(x19,y19,'c--',markersize=7, label='2.5% err, median')

#Make legend
legend = plt.legend(loc='upper right',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')

#Add title
plt.title('MH370 Final Locations -- Where to Search', fontsize=30)

#Show below
#plt.show()
file = 'plot_ln215.svg'
savefig(plt, file)

################################################################################################################################################################################################                
