import React, { useState } from 'react';
import { ArrowRight, Activity, BarChart3, Map, FileBarChart, HeartPulse, Info } from 'lucide-react';

function App() {
  const [selectedSection, setSelectedSection] = useState('overview');

  const sections = {
    overview: {
      title: 'AfyaScope Kenya Dashboard',
      icon: <Activity size={24} />,
      description: 'Interactive visualization platform for exploring HIV/STI data in Kenya',
      action: 'Launch Dashboard',
      link: 'http://localhost:8501', // Streamlit runs on port 8501 by default
    },
    data: {
      title: 'Data Sources',
      icon: <FileBarChart size={24} />,
      description: 'Comprehensive HIV/STI datasets from Kenya covering prevalence, indicators, and UNAIDS facts',
      content: (
        <div className="mt-4">
          <h3 className="text-xl font-semibold mb-2">Available Datasets</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">prevalence.csv</h4>
              <p className="text-sm text-gray-600 mt-1">HIV and STI prevalence rates by county and year</p>
              <p className="text-xs text-gray-500 mt-2">Columns: County, Year, HIV Prevalence (%), STI Rate, ART Coverage (%)</p>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">indicators.csv</h4>
              <p className="text-sm text-gray-600 mt-1">HIV/STI indicators by county and year</p>
              <p className="text-xs text-gray-500 mt-2">Columns: County, Year, Indicator, Value</p>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">unaids_facts.csv</h4>
              <p className="text-sm text-gray-600 mt-1">Extracted summaries from UNAIDS fact sheets</p>
              <p className="text-xs text-gray-500 mt-2">Columns: Year, Key Facts, Highlights</p>
            </div>
          </div>
        </div>
      ),
    },
    visualizations: {
      title: 'Visualizations',
      icon: <BarChart3 size={24} />,
      description: 'Interactive charts, maps, and graphs showing HIV/STI trends across Kenya',
      content: (
        <div className="mt-4">
          <h3 className="text-xl font-semibold mb-2">Available Visualizations</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">Choropleth Maps</h4>
              <p className="text-sm text-gray-600 mt-1">Geographic visualization of HIV prevalence across counties</p>
              <img 
                src="https://images.pexels.com/photos/7504837/pexels-photo-7504837.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" 
                alt="Sample choropleth map"
                className="w-full h-40 object-cover rounded mt-2"
              />
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">Trend Analysis</h4>
              <p className="text-sm text-gray-600 mt-1">Time series charts showing HIV/STI trends over time</p>
              <img 
                src="https://images.pexels.com/photos/7947551/pexels-photo-7947551.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" 
                alt="Sample trend chart"
                className="w-full h-40 object-cover rounded mt-2"
              />
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">County Comparisons</h4>
              <p className="text-sm text-gray-600 mt-1">Bar charts for comparing HIV prevalence across counties</p>
              <img 
                src="https://images.pexels.com/photos/590022/pexels-photo-590022.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" 
                alt="Sample bar chart"
                className="w-full h-40 object-cover rounded mt-2"
              />
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">Predictive Models</h4>
              <p className="text-sm text-gray-600 mt-1">Visualizations of forecasted HIV/STI trends</p>
              <img 
                src="https://images.pexels.com/photos/7947420/pexels-photo-7947420.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" 
                alt="Sample prediction chart"
                className="w-full h-40 object-cover rounded mt-2"
              />
            </div>
          </div>
        </div>
      ),
    },
    analytics: {
      title: 'Analytics Models',
      icon: <HeartPulse size={24} />,
      description: 'Advanced data science models for trend analysis and predictions',
      content: (
        <div className="mt-4">
          <h3 className="text-xl font-semibold mb-2">Available Models</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">Time Series Forecasting</h4>
              <p className="text-sm text-gray-600 mt-1">Prophet-based models for forecasting future HIV/STI trends</p>
              <ul className="text-xs text-gray-500 mt-2 list-disc pl-4">
                <li>5-year forecasts for each county</li>
                <li>Confidence intervals for predictions</li>
                <li>Trend and seasonality decomposition</li>
              </ul>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">Risk Classification</h4>
              <p className="text-sm text-gray-600 mt-1">Machine learning models to identify high-risk areas</p>
              <ul className="text-xs text-gray-500 mt-2 list-disc pl-4">
                <li>Random Forest classification algorithm</li>
                <li>Feature importance analysis</li>
                <li>Model performance metrics</li>
              </ul>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">Trend Analysis</h4>
              <p className="text-sm text-gray-600 mt-1">Statistical analysis of historical trends and patterns</p>
              <ul className="text-xs text-gray-500 mt-2 list-disc pl-4">
                <li>Year-over-year change calculation</li>
                <li>Regional comparisons</li>
                <li>Correlation analysis</li>
              </ul>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h4 className="font-medium text-teal-700">Geospatial Analysis</h4>
              <p className="text-sm text-gray-600 mt-1">Spatial analysis of HIV/STI distribution</p>
              <ul className="text-xs text-gray-500 mt-2 list-disc pl-4">
                <li>County-level choropleth maps</li>
                <li>Regional clustering</li>
                <li>Hotspot identification</li>
              </ul>
            </div>
          </div>
        </div>
      ),
    },
    about: {
      title: 'About the Project',
      icon: <Info size={24} />,
      description: 'Learn more about the AfyaScope Kenya project and its goals',
      content: (
        <div className="mt-4">
          <h3 className="text-xl font-semibold mb-2">Project Overview</h3>
          <p className="text-gray-700 leading-relaxed">
            AfyaScope Kenya is a data science project that analyzes and predicts HIV/AIDS and STI trends in Kenya using real-world data. 
            The project aims to present the results in an understandable way for the general public, healthcare workers, and policymakers.
          </p>
          
          <h3 className="text-xl font-semibold mt-6 mb-2">Features</h3>
          <ul className="list-disc pl-5 text-gray-700">
            <li>Data cleaning and integration of multiple HIV/STI datasets</li>
            <li>Exploratory data analysis with visualizations</li>
            <li>Geospatial mapping of prevalence rates across counties</li>
            <li>Time series forecasting of HIV/STI trends</li>
            <li>Interactive Streamlit dashboard for data exploration</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-2">Technologies Used</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
            <div className="bg-gray-100 p-2 rounded text-center text-sm">Python</div>
            <div className="bg-gray-100 p-2 rounded text-center text-sm">Pandas</div>
            <div className="bg-gray-100 p-2 rounded text-center text-sm">Matplotlib</div>
            <div className="bg-gray-100 p-2 rounded text-center text-sm">Seaborn</div>
            <div className="bg-gray-100 p-2 rounded text-center text-sm">Scikit-learn</div>
            <div className="bg-gray-100 p-2 rounded text-center text-sm">Prophet</div>
            <div className="bg-gray-100 p-2 rounded text-center text-sm">Streamlit</div>
            <div className="bg-gray-100 p-2 rounded text-center text-sm">Folium</div>
          </div>

          <h3 className="text-xl font-semibold mt-6 mb-2">Project Goals</h3>
          <p className="text-gray-700 leading-relaxed">
            The primary goal of AfyaScope Kenya is to provide actionable insights that can help in the fight against HIV/AIDS and STIs in Kenya. 
            By identifying trends, patterns, and high-risk areas, we aim to support evidence-based decision-making for healthcare programs and interventions.
          </p>
        </div>
      ),
    },
  };

  const selectedContent = sections[selectedSection as keyof typeof sections];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-teal-700 text-white shadow-md">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center">
            <HeartPulse size={28} className="mr-2" />
            <h1 className="text-xl font-bold">AfyaScope Kenya</h1>
          </div>
          <div className="text-sm">HIV &amp; STI Trends and Predictions</div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <div className="md:col-span-1">
            <div className="bg-white rounded-lg shadow overflow-hidden">
              <div className="p-4 bg-teal-50 border-b border-gray-200">
                <h2 className="text-lg font-medium text-teal-800">Navigation</h2>
              </div>
              <nav className="p-2">
                {Object.entries(sections).map(([key, section]) => (
                  <button
                    key={key}
                    onClick={() => setSelectedSection(key)}
                    className={`w-full text-left px-4 py-3 rounded-md flex items-center transition-colors ${
                      selectedSection === key ? 'bg-teal-50 text-teal-700' : 'hover:bg-gray-50'
                    }`}
                  >
                    <span className="mr-3">{section.icon}</span>
                    <span>{section.title}</span>
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="md:col-span-3">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-teal-800 flex items-center">
                {selectedContent.icon && <span className="mr-3">{selectedContent.icon}</span>}
                {selectedContent.title}
              </h2>
              <p className="text-gray-600 mt-2">{selectedContent.description}</p>
              
              {selectedContent.link && (
                <a 
                  href={selectedContent.link} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center mt-4 px-6 py-2 bg-teal-600 text-white font-medium rounded-md shadow-sm hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 transition-colors"
                >
                  {selectedContent.action} <ArrowRight size={18} className="ml-2" />
                </a>
              )}
              
              {selectedContent.content && selectedContent.content}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 text-white mt-12 py-8">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between">
            <div className="mb-6 md:mb-0">
              <h3 className="text-lg font-semibold mb-2 flex items-center">
                <HeartPulse size={20} className="mr-2" /> AfyaScope Kenya
              </h3>
              <p className="text-gray-400 text-sm">
                Data-driven insights for HIV/STI trends in Kenya
              </p>
            </div>
            <div className="grid grid-cols-2 gap-8 sm:grid-cols-3">
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider">Resources</h3>
                <ul className="mt-2 space-y-1">
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">Documentation</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">API</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">Datasets</a></li>
                </ul>
              </div>
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider">Community</h3>
                <ul className="mt-2 space-y-1">
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">GitHub</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">Partners</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">Contact</a></li>
                </ul>
              </div>
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider">Legal</h3>
                <ul className="mt-2 space-y-1">
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">Privacy</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">Terms</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-white text-sm">License</a></li>
                </ul>
              </div>
            </div>
          </div>
          <div className="mt-8 border-t border-gray-700 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm">© 2025 AfyaScope Kenya. All rights reserved.</p>
            <div className="mt-4 md:mt-0 flex space-x-6">
              <a href="#" className="text-gray-400 hover:text-white">
                <span className="sr-only">Twitter</span>
                <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
                </svg>
              </a>
              <a href="#" className="text-gray-400 hover:text-white">
                <span className="sr-only">GitHub</span>
                <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;