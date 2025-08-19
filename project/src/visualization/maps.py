"""
Module for creating maps and geographical visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union
import os


class MapGenerator:
    """Class for generating maps and geographical visualizations."""
    
    def __init__(self, kenya_geojson_path: Optional[str] = None):
        """
        Initialize the MapGenerator.
        
        Args:
            kenya_geojson_path: Path to Kenya counties GeoJSON file (if None, will download)
        """
        self.kenya_geojson_path = kenya_geojson_path
        self.kenya_gdf = self._load_kenya_geodata()
        
        # Kenya approximate centroid
        self.kenya_center = [-1.2921, 36.8219]  # Nairobi coordinates as center
        
    def _load_kenya_geodata(self) -> gpd.GeoDataFrame:
        """
        Load Kenya GeoJSON data for counties.
        
        Returns:
            GeoDataFrame with Kenya counties geometries
        """
        try:
            if self.kenya_geojson_path and os.path.exists(self.kenya_geojson_path):
                gdf = gpd.read_file(self.kenya_geojson_path)
            else:
                # If no file provided, use a built-in dataset or placeholder
                # For a real project, you would download a proper GeoJSON
                # This is just a placeholder solution
                print("No Kenya GeoJSON provided. Using simplified polygon data.")
                
                # Create a simple rectangular geometry for demonstration
                from shapely.geometry import Polygon
                
                # Simple rectangular representations of counties
                counties = []
                geometries = []
                
                # Define Kenya counties with placeholder geometries
                for i, county in enumerate(["Nairobi", "Mombasa", "Kisumu", "Nakuru"]):
                    # Create a simple polygon (this is just for demonstration)
                    x_offset = (i % 2) * 0.5
                    y_offset = (i // 2) * 0.5
                    poly = Polygon([
                        (36.5 + x_offset, -1.0 + y_offset),
                        (37.0 + x_offset, -1.0 + y_offset),
                        (37.0 + x_offset, -1.5 + y_offset),
                        (36.5 + x_offset, -1.5 + y_offset)
                    ])
                    counties.append(county)
                    geometries.append(poly)
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    {'county': counties, 'geometry': geometries},
                    crs="EPSG:4326"
                )
                
                # Save for future use
                os.makedirs("data/geodata", exist_ok=True)
                gdf.to_file("data/geodata/kenya_counties_placeholder.geojson", driver="GeoJSON")
            
            return gdf
        
        except Exception as e:
            print(f"Error loading Kenya geodata: {e}")
            # Return empty GeoDataFrame if loading fails
            return gpd.GeoDataFrame()
    
    def create_choropleth_map(self,
                             df: pd.DataFrame,
                             value_col: str,
                             year: int,
                             title: str,
                             colormap: str = 'YlOrRd',
                             save_path: Optional[str] = None) -> folium.Map:
        """
        Create a choropleth map of Kenya counties with data values.
        
        Args:
            df: DataFrame containing county data
            value_col: Column containing values to visualize
            year: Year to filter data for
            title: Map title
            colormap: Colormap to use for values
            save_path: Path to save the HTML map
            
        Returns:
            Folium Map object
        """
        if df.empty or 'county' not in df.columns or value_col not in df.columns:
            return folium.Map(location=self.kenya_center, zoom_start=7)
            
        # Filter data for the specified year
        year_df = df[df['year'] == year].copy()
        
        if year_df.empty:
            return folium.Map(location=self.kenya_center, zoom_start=7)
            
        # Create the base map
        m = folium.Map(location=self.kenya_center, zoom_start=7,
                      tiles="cartodbpositron")
        
        # Add title
        title_html = f'''
            <h3 align="center" style="font-size: 16px">{title}</h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Check if we have proper geodata
        if not self.kenya_gdf.empty:
            # Merge data with geodata
            merged_gdf = self.kenya_gdf.copy()
            
            # Prepare county names for merging (case-insensitive)
            merged_gdf['county_lower'] = merged_gdf['county'].str.lower()
            year_df['county_lower'] = year_df['county'].str.lower()
            
            # Merge by county
            merged_gdf = merged_gdf.merge(
                year_df[['county_lower', value_col]],
                on='county_lower',
                how='left'
            )
            
            # Calculate bounds for colormap
            vmin = merged_gdf[value_col].min()
            vmax = merged_gdf[value_col].max()
            
            # Create choropleth
            folium.Choropleth(
                geo_data=merged_gdf.__geo_interface__,
                name="choropleth",
                data=year_df,
                columns=["county", value_col],
                key_on="feature.properties.county",
                fill_color=colormap,
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=value_col.replace('_', ' ').title(),
                highlight=True,
                reset=True
            ).add_to(m)
            
            # Add hover functionality
            style_function = lambda x: {'fillColor': '#ffffff', 
                                       'color': '#000000', 
                                       'fillOpacity': 0.1, 
                                       'weight': 0.5}
            highlight_function = lambda x: {'fillColor': '#000000', 
                                          'color': '#000000', 
                                          'fillOpacity': 0.5, 
                                          'weight': 0.7}
            
            # Add tooltips
            counties_tooltip = folium.features.GeoJson(
                merged_gdf,
                style_function=style_function,
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['county', value_col],
                    aliases=['County:', f'{value_col.replace("_", " ").title()}:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            )
            m.add_child(counties_tooltip)
            
        else:
            # If we don't have proper geodata, create markers instead
            marker_cluster = MarkerCluster().add_to(m)
            
            for _, row in year_df.iterrows():
                # We don't have actual coordinates, so we'll create some from county name hash
                # This is just for demonstration - in reality you'd use actual coordinates
                county_hash = hash(row['county'])
                lat_offset = (county_hash % 100) / 100
                lon_offset = (county_hash % 50) / 100
                
                lat = self.kenya_center[0] + lat_offset
                lon = self.kenya_center[1] + lon_offset
                
                popup_text = f"<b>{row['county']}</b><br>{value_col.replace('_', ' ').title()}: {row[value_col]:.2f}"
                
                folium.Marker(
                    location=[lat, lon],
                    popup=popup_text,
                    tooltip=row['county']
                ).add_to(marker_cluster)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save to HTML file if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            m.save(save_path)
            print(f"Map saved to {save_path}")
        
        return m
    
    def create_bubble_map(self,
                         df: pd.DataFrame,
                         value_col: str,
                         year: int,
                         title: str,
                         color: str = '#3186cc',
                         save_path: Optional[str] = None) -> folium.Map:
        """
        Create a bubble map of Kenya counties with data values shown as bubbles.
        
        Args:
            df: DataFrame containing county data
            value_col: Column containing values to visualize as bubble size
            year: Year to filter data for
            title: Map title
            color: Color for the bubbles
            save_path: Path to save the HTML map
            
        Returns:
            Folium Map object
        """
        if df.empty or 'county' not in df.columns or value_col not in df.columns:
            return folium.Map(location=self.kenya_center, zoom_start=7)
            
        # Filter data for the specified year
        year_df = df[df['year'] == year].copy()
        
        if year_df.empty:
            return folium.Map(location=self.kenya_center, zoom_start=7)
            
        # Create the base map
        m = folium.Map(location=self.kenya_center, zoom_start=7,
                      tiles="cartodbpositron")
        
        # Add title
        title_html = f'''
            <h3 align="center" style="font-size: 16px">{title}</h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Check if we have proper geodata
        if not self.kenya_gdf.empty:
            # Add county boundaries
            folium.GeoJson(
                self.kenya_gdf,
                name="Counties",
                style_function=lambda x: {
                    'fillColor': '#ffffff',
                    'color': '#000000',
                    'fillOpacity': 0.1,
                    'weight': 0.5
                }
            ).add_to(m)
            
            # Merge data with geodata
            merged_gdf = self.kenya_gdf.copy()
            
            # Prepare county names for merging (case-insensitive)
            merged_gdf['county_lower'] = merged_gdf['county'].str.lower()
            year_df['county_lower'] = year_df['county'].str.lower()
            
            # Merge by county
            merged_gdf = merged_gdf.merge(
                year_df[['county_lower', value_col]],
                on='county_lower',
                how='left'
            )
            
            # Calculate centroids for bubble placement
            merged_gdf['centroid'] = merged_gdf.geometry.centroid
            merged_gdf['lon'] = merged_gdf.centroid.x
            merged_gdf['lat'] = merged_gdf.centroid.y
            
            # Normalize values for circle size
            min_val = merged_gdf[value_col].min()
            max_val = merged_gdf[value_col].max()
            merged_gdf['circle_size'] = ((merged_gdf[value_col] - min_val) / (max_val - min_val)) * 40 + 5
            
            # Add bubbles
            for _, row in merged_gdf.iterrows():
                if pd.notna(row[value_col]):
                    folium.Circle(
                        location=[row['lat'], row['lon']],
                        radius=row['circle_size'] * 1000,  # Convert to meters
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.6,
                        tooltip=f"{row['county']}: {row[value_col]:.2f}",
                        popup=f"<b>{row['county']}</b><br>{value_col.replace('_', ' ').title()}: {row[value_col]:.2f}"
                    ).add_to(m)
            
        else:
            # If we don't have proper geodata, create markers instead
            for _, row in year_df.iterrows():
                # Generate pseudo-coordinates
                county_hash = hash(row['county'])
                lat_offset = (county_hash % 100) / 100
                lon_offset = (county_hash % 50) / 100
                
                lat = self.kenya_center[0] + lat_offset
                lon = self.kenya_center[1] + lon_offset
                
                # Normalize for circle size
                min_val = year_df[value_col].min()
                max_val = year_df[value_col].max()
                circle_size = ((row[value_col] - min_val) / (max_val - min_val)) * 40 + 5
                
                folium.Circle(
                    location=[lat, lon],
                    radius=circle_size * 1000,  # Convert to meters
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    tooltip=f"{row['county']}: {row[value_col]:.2f}",
                    popup=f"<b>{row['county']}</b><br>{value_col.replace('_', ' ').title()}: {row[value_col]:.2f}"
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add a legend
        legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 150px; height: 90px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color:white;
                        padding: 5px;
                        border-radius: 5px;">
            <p style="margin-top: 0;"><b>Legend</b></p>
            <div style="display: flex; align-items: center;">
                <div style="background-color: ''' + color + '''; 
                            width: 15px; height: 15px; 
                            border-radius: 50%;
                            margin-right: 5px;"></div>
                <span>''' + value_col.replace('_', ' ').title() + '''</span>
            </div>
            <p style="margin-bottom: 0;"><i>Bubble size represents value</i></p>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save to HTML file if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            m.save(save_path)
            print(f"Map saved to {save_path}")
        
        return m
    
    def create_dual_map(self,
                       df: pd.DataFrame,
                       value_col1: str,
                       value_col2: str,
                       year: int,
                       title: str,
                       save_path: Optional[str] = None) -> folium.Map:
        """
        Create a side-by-side map comparing two indicators for the same year.
        
        Args:
            df: DataFrame containing county data
            value_col1: First column to visualize
            value_col2: Second column to visualize
            year: Year to filter data for
            title: Map title
            save_path: Path to save the HTML map
            
        Returns:
            Folium Map object with side-by-side maps
        """
        if df.empty or 'county' not in df.columns:
            return folium.Map(location=self.kenya_center, zoom_start=7)
            
        # Filter data for the specified year
        year_df = df[df['year'] == year].copy()
        
        if year_df.empty:
            return folium.Map(location=self.kenya_center, zoom_start=7)
            
        # Create a map with two side-by-side panels
        m = folium.plugins.DualMap(
            location=self.kenya_center,
            zoom_start=7,
            tiles="cartodbpositron"
        )
        
        # Add title
        title_html = f'''
            <h3 align="center" style="font-size: 16px">{title}</h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Check if we have proper geodata
        if not self.kenya_gdf.empty:
            # Merge data with geodata
            merged_gdf = self.kenya_gdf.copy()
            
            # Prepare county names for merging (case-insensitive)
            merged_gdf['county_lower'] = merged_gdf['county'].str.lower()
            year_df['county_lower'] = year_df['county'].str.lower()
            
            # Merge by county
            merged_gdf = merged_gdf.merge(
                year_df[['county_lower', value_col1, value_col2]],
                on='county_lower',
                how='left'
            )
            
            # Map 1: First value
            folium.Choropleth(
                geo_data=merged_gdf.__geo_interface__,
                name=value_col1,
                data=year_df,
                columns=["county", value_col1],
                key_on="feature.properties.county",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=value_col1.replace('_', ' ').title(),
                highlight=True
            ).add_to(m.m1)
            
            # Map 2: Second value
            folium.Choropleth(
                geo_data=merged_gdf.__geo_interface__,
                name=value_col2,
                data=year_df,
                columns=["county", value_col2],
                key_on="feature.properties.county",
                fill_color="Blues",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=value_col2.replace('_', ' ').title(),
                highlight=True
            ).add_to(m.m2)
            
            # Add tooltips to both maps
            style_function = lambda x: {'fillColor': '#ffffff', 
                                       'color': '#000000', 
                                       'fillOpacity': 0.1, 
                                       'weight': 0.5}
            highlight_function = lambda x: {'fillColor': '#000000', 
                                          'color': '#000000', 
                                          'fillOpacity': 0.5, 
                                          'weight': 0.7}
            
            # Add tooltips to map 1
            tooltip1 = folium.features.GeoJson(
                merged_gdf,
                style_function=style_function,
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['county', value_col1],
                    aliases=['County:', f'{value_col1.replace("_", " ").title()}:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            )
            m.m1.add_child(tooltip1)
            
            # Add tooltips to map 2
            tooltip2 = folium.features.GeoJson(
                merged_gdf,
                style_function=style_function,
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['county', value_col2],
                    aliases=['County:', f'{value_col2.replace("_", " ").title()}:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            )
            m.m2.add_child(tooltip2)
            
        else:
            # If we don't have proper geodata, create markers instead
            for _, row in year_df.iterrows():
                # Generate pseudo-coordinates
                county_hash = hash(row['county'])
                lat_offset = (county_hash % 100) / 100
                lon_offset = (county_hash % 50) / 100
                
                lat = self.kenya_center[0] + lat_offset
                lon = self.kenya_center[1] + lon_offset
                
                # Add to first map
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{row['county']}</b><br>{value_col1.replace('_', ' ').title()}: {row[value_col1]:.2f}",
                    tooltip=row['county'],
                    icon=folium.Icon(color='red')
                ).add_to(m.m1)
                
                # Add to second map
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{row['county']}</b><br>{value_col2.replace('_', ' ').title()}: {row[value_col2]:.2f}",
                    tooltip=row['county'],
                    icon=folium.Icon(color='blue')
                ).add_to(m.m2)
        
        # Add layer control to both maps
        folium.LayerControl().add_to(m.m1)
        folium.LayerControl().add_to(m.m2)
        
        # Add map titles
        title1_html = f'''
            <h4 align="center" style="font-size: 14px">{value_col1.replace('_', ' ').title()}</h4>
        '''
        title2_html = f'''
            <h4 align="center" style="font-size: 14px">{value_col2.replace('_', ' ').title()}</h4>
        '''
        m.m1.get_root().html.add_child(folium.Element(title1_html))
        m.m2.get_root().html.add_child(folium.Element(title2_html))
        
        # Save to HTML file if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            m.save(save_path)
            print(f"Map saved to {save_path}")
        
        return m


if __name__ == "__main__":
    # This code runs when the script is executed directly
    import os
    from src.data.data_loader import DataLoader
    from src.data.data_cleaner import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    cleaner = DataCleaner()
    
    raw_datasets = loader.load_all_datasets()
    cleaned_datasets = cleaner.clean_all_datasets(raw_datasets)
    
    # Get the prevalence dataset
    prevalence_df = cleaned_datasets.get('prevalence', pd.DataFrame())
    
    if not prevalence_df.empty:
        # Initialize map generator
        map_gen = MapGenerator()
        
        # Example: Create choropleth map
        map_html = map_gen.create_choropleth_map(
            prevalence_df,
            'hiv_prevalence',
            2020,  # Use a year that exists in your data
            "HIV Prevalence by County (2020)",
            save_path="data/visualizations/hiv_prevalence_map_2020.html"
        )
        
        # Example: Create bubble map
        map_html = map_gen.create_bubble_map(
            prevalence_df,
            'sti_rate',
            2020,  # Use a year that exists in your data
            "STI Rate by County (2020)",
            color="#ff7f0e",
            save_path="data/visualizations/sti_rate_map_2020.html"
        )