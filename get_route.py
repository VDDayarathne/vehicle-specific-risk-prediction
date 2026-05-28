import requests
import json
import math

# Coordinates: Start near Kadugannawa (e.g. Captain Dawson Tower / Balumgala area) and end in Mawanella Town.
# Start: Kadugannawa (7.2583, 80.5218)
# End: Mawanella (7.2512, 80.4440)
start_lat, start_lon = 7.2583, 80.5218
end_lat, end_lon = 7.2512, 80.4440

url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson&steps=true"

try:
    print(f"Fetching route from: {url}")
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    data = res.json()
    
    if "routes" in data and len(data["routes"]) > 0:
        route = data["routes"][0]
        geometry = route["geometry"]
        coordinates = geometry["coordinates"] # List of [lon, lat]
        
        # OSRM returns [lon, lat]. Let's convert to [lat, lon] for Leaflet
        lat_lons = [[pt[1], pt[0]] for pt in coordinates]
        print(f"Successfully fetched {len(lat_lons)} high-resolution road coordinates!")
        
        # Calculate heading / bearing change between successive coordinates to detect sharp bends/curves
        bends = []
        
        def calculate_bearing(lat1, lon1, lat2, lon2):
            dLon = math.radians(lon2 - lon1)
            lat1 = math.radians(lat1)
            lat2 = math.radians(lat2)
            y = math.sin(dLon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
            brng = math.atan2(y, x)
            return (math.degrees(brng) + 360) % 360

        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000 # meters
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c

        # Compute cumulative distance along the road
        distances = [0.0]
        cum_dist = 0.0
        for i in range(1, len(lat_lons)):
            d = haversine_distance(lat_lons[i-1][0], lat_lons[i-1][1], lat_lons[i][0], lat_lons[i][1])
            cum_dist += d
            distances.append(cum_dist)

        # Detect sharp bends
        # We can look at three points: P_{i-k}, P_i, P_{i+k} to compute heading change over a window
        window = 3
        bend_points = []
        for i in range(window, len(lat_lons) - window):
            lat_prev, lon_prev = lat_lons[i - window]
            lat_curr, lon_curr = lat_lons[i]
            lat_next, lon_next = lat_lons[i + window]
            
            b1 = calculate_bearing(lat_prev, lon_prev, lat_curr, lon_curr)
            b2 = calculate_bearing(lat_curr, lon_curr, lat_next, lon_next)
            
            diff = (b2 - b1 + 180) % 360 - 180
            abs_diff = abs(diff)
            
            if abs_diff > 25: # Significant turn (heading change > 25 degrees over 30-50m)
                bend_points.append({
                    "index": i,
                    "lat": lat_curr,
                    "lon": lon_curr,
                    "angle_diff": abs_diff,
                    "dist_m": distances[i]
                })
        
        # Filter local maxima to find distinct sharp bends
        distinct_bends = []
        bend_points.sort(key=lambda x: x["dist_m"])
        j = 0
        while j < len(bend_points):
            curr = bend_points[j]
            # Find all bend points within 150m and take the one with the maximum angle change
            cluster = [curr]
            k = j + 1
            while k < len(bend_points) and (bend_points[k]["dist_m"] - curr["dist_m"]) < 150:
                cluster.append(bend_points[k])
                k += 1
            best = max(cluster, key=lambda x: x["angle_diff"])
            distinct_bends.append(best)
            j = k
            
        print(f"Detected {len(distinct_bends)} distinct sharp bends along the route:")
        
        # Write to JSON file
        result = {
            "road_path": lat_lons,
            "bends": distinct_bends,
            "total_distance_m": cum_dist
        }
        with open("C:/Users/ASUS/.gemini/antigravity/brain/5b8f59cb-2ddc-4425-9f4b-48a7c2d7722e/route_details.json", "w") as out:
            json.dump(result, out, indent=2)
        print("Saved successfully!")
            
    else:
        print("No route found in response!")
except Exception as e:
    print(f"Error occurred: {e}")
