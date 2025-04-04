// f1-prediction-app.js
// React component for F1 qualifying prediction

const { useState, useEffect } = React;
const { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} = Recharts;

// Base API URL - change this to match your backend location
const API_BASE_URL = "http://localhost:5000";

// Team colors for visualization
const teamColors = {
  'VER': '#3671C6', // Red Bull
  'PER': '#3671C6',
  'HAM': '#27F4D2', // Mercedes
  'RUS': '#27F4D2',
  'LEC': '#F91536', // Ferrari
  'SAI': '#F91536',
  'NOR': '#FF8700', // McLaren
  'PIA': '#FF8700',
  'ALO': '#358C75', // Aston Martin
  'STR': '#358C75',
  'GAS': '#2293D1', // Alpine
  'OCO': '#2293D1',
  'ALB': '#37BEDD', // Williams
  'SAR': '#37BEDD',
  'TSU': '#5E8FAA', // RB (AlphaTauri)
  'RIC': '#5E8FAA',
  'BOT': '#C92D4B', // Sauber (Alfa Romeo)
  'ZHO': '#C92D4B',
  'MAG': '#B6BABD', // Haas
  'HUL': '#B6BABD'
};

// Additional circuit data for visualization
const circuitCharacteristics = {
  'Monaco': { sectors: [1.2, 0.9, 1.3], difficulty: 9.5, overtaking: 2 },
  'Silverstone': { sectors: [1.1, 1.4, 0.9], difficulty: 7.5, overtaking: 7 },
  'Monza': { sectors: [1.5, 1.3, 1.4], difficulty: 6.5, overtaking: 8.5 },
  'Spa': { sectors: [1.3, 1.5, 1.2], difficulty: 8, overtaking: 8 },
  'Suzuka': { sectors: [1.1, 1.2, 1.0], difficulty: 8.5, overtaking: 6 },
  'Singapore': { sectors: [1.1, 1.0, 1.2], difficulty: 9, overtaking: 3 },
  'Barcelona': { sectors: [1.0, 1.1, 0.9], difficulty: 7, overtaking: 5 },
  'Melbourne': { sectors: [1.0, 1.2, 0.9], difficulty: 7, overtaking: 6 },
  'Montreal': { sectors: [1.1, 0.9, 1.0], difficulty: 7.5, overtaking: 7.5 },
  'Bahrain': { sectors: [1.1, 1.0, 1.2], difficulty: 6.5, overtaking: 7.5 },
  'Hungaroring': { sectors: [0.9, 1.1, 1.0], difficulty: 8, overtaking: 4 },
  'Zandvoort': { sectors: [1.0, 1.2, 0.9], difficulty: 7.5, overtaking: 5 },
  'Jeddah': { sectors: [1.4, 1.3, 1.2], difficulty: 9, overtaking: 6 },
  'Miami': { sectors: [1.1, 1.2, 1.0], difficulty: 7, overtaking: 6.5 },
  'Las Vegas': { sectors: [1.3, 1.4, 1.2], difficulty: 7.5, overtaking: 7 },
  'Baku': { sectors: [1.2, 1.1, 1.3], difficulty: 8.5, overtaking: 7 },
  'Red Bull Ring': { sectors: [1.0, 1.1, 0.9], difficulty: 6.5, overtaking: 7 },
  'Imola': { sectors: [1.0, 1.1, 0.9], difficulty: 7.5, overtaking: 5.5 },
  'Interlagos': { sectors: [1.0, 1.1, 1.2], difficulty: 7, overtaking: 8 },
  'Qatar': { sectors: [1.1, 1.2, 1.0], difficulty: 7.5, overtaking: 6 },
  'COTA': { sectors: [1.2, 1.1, 1.3], difficulty: 7.5, overtaking: 7 },
  'Mexico City': { sectors: [1.1, 1.0, 1.2], difficulty: 7, overtaking: 6 },
  'Abu Dhabi': { sectors: [1.1, 1.2, 1.0], difficulty: 7, overtaking: 6 },
  'Saudi Arabia': { sectors: [1.3, 1.2, 1.4], difficulty: 8.5, overtaking: 6.5 }
};

// Icon component for Flag
const FlagIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"></path>
    <line x1="4" y1="22" x2="4" y2="15"></line>
  </svg>
);

// Icon component for Trophy
const TrophyIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-yellow-500">
    <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path>
    <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path>
    <path d="M4 22h16"></path>
    <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"></path>
    <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"></path>
    <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"></path>
  </svg>
);

// Icon component for Car
const CarIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
    <path d="M14 16H9m10 0h3v-3.15a1 1 0 0 0-.84-.99L16 11l-2.7-3.6a1 1 0 0 0-.8-.4H5.24a2 2 0 0 0-1.8 1.1l-.8 1.63A6 6 0 0 0 2 12.42V16h2"></path>
    <circle cx="6.5" cy="16.5" r="2.5"></circle>
    <circle cx="16.5" cy="16.5" r="2.5"></circle>
  </svg>
);

// Icon component for Alert Triangle
const AlertTriangleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
    <path d="M12 9v4"></path>
    <path d="M12 17h.01"></path>
  </svg>
);

// Icon component for Info
const InfoIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <path d="M12 16v-4"></path>
    <path d="M12 8h.01"></path>
  </svg>
);

// Icon component for RefreshCw (loading spinner)
const RefreshCwIcon = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
    <path d="M21 3v5h-5"></path>
    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
    <path d="M3 21v-5h5"></path>
  </svg>
);

const F1PredictionApp = () => {
  const [selectedCircuit, setSelectedCircuit] = useState('');
  const [predictionResults, setPredictionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [showCircuitStats, setShowCircuitStats] = useState(false);
  const [circuits, setCircuits] = useState([]);
  const [loadingCircuits, setLoadingCircuits] = useState(true);
  const [selectedCircuitData, setSelectedCircuitData] = useState(null);

  // Load circuits from API on component mount
  useEffect(() => {
    const fetchCircuits = async () => {
      setLoadingCircuits(true);
      try {
        const response = await fetch(`${API_BASE_URL}/api/circuits`);
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }
        const data = await response.json();
        setCircuits(data.circuits || []);
      } catch (err) {
        console.error("Error fetching circuits:", err);
        setError("Failed to load circuits. Please refresh the page.");
      } finally {
        setLoadingCircuits(false);
      }
    };

    fetchCircuits();
  }, []);

  // Update circuit data when circuit is selected
  useEffect(() => {
    if (selectedCircuit && circuits.length > 0) {
      const circuitObj = circuits.find(c => c.id === selectedCircuit);
      if (circuitObj) {
        // Try to match with circuit characteristics
        const circuitName = circuitObj.name;
        if (circuitCharacteristics[circuitName]) {
          setSelectedCircuitData({
            name: circuitName,
            ...circuitCharacteristics[circuitName]
          });
        } else {
          // Default values if no match found
          setSelectedCircuitData({
            name: circuitName,
            sectors: [1.0, 1.0, 1.0],
            difficulty: 7.0,
            overtaking: 6.0
          });
        }
      }
    } else {
      setSelectedCircuitData(null);
    }
  }, [selectedCircuit, circuits]);

  // Get prediction from the backend API
  const getPrediction = async (circuit) => {
    setLoading(true);
    setError(null);
    
    try {
      // Make API call to your Python backend
      const response = await fetch(`${API_BASE_URL}/api/predict?circuit=${encodeURIComponent(circuit)}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error ${response.status}`);
      }
      
      const predictionData = await response.json();
      
      // Handle possible error from backend
      if (predictionData.error) {
        throw new Error(predictionData.error);
      }
      
      setPredictionResults(predictionData);
      setConfidence(predictionData.confidence_score);
      
      // Get circuit type from the response if available
      if (predictionData.circuit_info && predictionData.circuit_info.type) {
        const circuitType = predictionData.circuit_info.type;
        console.log(`Circuit type from API: ${circuitType}`);
      }
    } catch (err) {
      setError(`Failed to get prediction: ${err.message}`);
      console.error("Prediction error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handlePrediction = () => {
    if (selectedCircuit) {
      getPrediction(selectedCircuit);
    } else {
      setError("Please select a circuit first");
    }
  };

  // Format probability as percentage
  const formatProbability = (prob) => {
    return `${(prob * 100).toFixed(1)}%`;
  };

  // Get driver team for color coding
  const getDriverTeam = (driverCode) => {
    return teamColors[driverCode] || "#999999";
  };

  // Get circuit type icon
  const getCircuitTypeIcon = (circuitInfo) => {
    if (!circuitInfo) return '🏁';
    const type = circuitInfo.type || 'technical';
    if (type === 'street') return '🏙️';
    if (type === 'high_speed') return '🚀';
    return '🔄';
  };

  const getCircuitTypeLabel = (circuitInfo) => {
    if (!circuitInfo) return 'Circuit';
    const type = circuitInfo.type || 'technical';
    if (type === 'street') return 'Street Circuit';
    if (type === 'high_speed') return 'High Speed Circuit';
    return 'Technical Circuit';
  };

  // Create chart data
  const createChartData = () => {
    if (!predictionResults) return [];
    
    return predictionResults.top5.map(driver => ({
      name: driver.driver_code,
      probability: Math.round(driver.probability * 100),
      fill: getDriverTeam(driver.driver_code)
    }));
  };

  return (
    <div className="flex flex-col bg-gray-50 p-6 rounded-lg shadow-lg w-full max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">F1 Qualifying Prediction</h1>
      
      {/* Circuit Selection */}
      <div className="flex flex-col md:flex-row items-center gap-4 mb-6">
        <div className="w-full md:w-2/3">
          <label htmlFor="circuit-select" className="block text-sm font-medium text-gray-700 mb-1">
            Select Circuit
          </label>
          {loadingCircuits ? (
            <div className="flex items-center p-2 border border-gray-300 rounded-md">
              <RefreshCwIcon className="animate-spin mr-2" />
              <span className="text-gray-500">Loading circuits...</span>
            </div>
          ) : (
            <select
              id="circuit-select"
              value={selectedCircuit}
              onChange={(e) => setSelectedCircuit(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">-- Select a Circuit --</option>
              {circuits.map((circuit) => (
                <option key={circuit.id} value={circuit.id}>
                  {circuit.name}
                </option>
              ))}
            </select>
          )}
        </div>
        <button
          onClick={handlePrediction}
          disabled={loading || !selectedCircuit || loadingCircuits}
          className="w-full md:w-1/3 mt-4 md:mt-7 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-md flex items-center justify-center gap-2 disabled:bg-gray-400"
        >
          {loading ? (
            <>
              <RefreshCwIcon className="animate-spin mr-1" />
              <span>Predicting...</span>
            </>
          ) : (
            <>
              <FlagIcon />
              <span>Predict Qualifying</span>
            </>
          )}
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6 rounded">
          <div className="flex items-center">
            <AlertTriangleIcon />
            <p className="ml-2">{error}</p>
          </div>
        </div>
      )}

      {/* Circuit Info */}
      {selectedCircuitData && (
        <div className="mb-6 bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-xl font-bold flex items-center gap-2">
                {selectedCircuitData.name} {getCircuitTypeIcon(predictionResults?.circuit_info)}
              </h2>
              <p className="text-gray-600 text-sm">{getCircuitTypeLabel(predictionResults?.circuit_info)}</p>
            </div>
            <button 
              onClick={() => setShowCircuitStats(!showCircuitStats)}
              className="text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              <InfoIcon />
              <span className="ml-1">{showCircuitStats ? 'Hide Details' : 'Show Details'}</span>
            </button>
          </div>
          
          {showCircuitStats && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-3 rounded border border-gray-200">
                <h3 className="font-semibold text-gray-700">Circuit Difficulty</h3>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                  <div 
                    className="bg-red-600 h-2.5 rounded-full" 
                    style={{ width: `${selectedCircuitData.difficulty * 10}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-600 mt-1">
                  {selectedCircuitData.difficulty}/10
                </p>
              </div>
              
              <div className="bg-gray-50 p-3 rounded border border-gray-200">
                <h3 className="font-semibold text-gray-700">Overtaking Potential</h3>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                  <div 
                    className="bg-green-600 h-2.5 rounded-full" 
                    style={{ width: `${selectedCircuitData.overtaking * 10}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-600 mt-1">
                  {selectedCircuitData.overtaking}/10
                </p>
              </div>
              
              <div className="bg-gray-50 p-3 rounded border border-gray-200">
                <h3 className="font-semibold text-gray-700">Sector Balance</h3>
                <div className="flex justify-between mt-2 gap-2">
                  {selectedCircuitData.sectors.map((value, index) => (
                    <div key={index} className="flex-1">
                      <div 
                        className={`w-full rounded-t-sm ${
                          index === 0 ? "bg-purple-500" : 
                          index === 1 ? "bg-blue-500" : "bg-green-500"
                        }`}
                        style={{ height: `${value * 30}px` }}
                      ></div>
                      <p className="text-xs text-center mt-1">S{index + 1}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Results Section */}
      {predictionResults && (
        <div className="bg-white rounded-lg shadow-md p-4 border border-gray-200">
          <h2 className="text-2xl font-bold mb-4 text-center">
            Predicted Qualifying Results for {predictionResults.circuit}
          </h2>
          <div className="text-center mb-4">
            <div className="inline-block bg-gray-100 px-4 py-2 rounded-full">
              <span className="font-semibold">Model Confidence:</span>
              <span className="ml-2">
                <span 
                  className={`inline-block h-3 w-24 rounded-full relative ${
                    confidence > 0.75 ? 'bg-green-200' : 
                    confidence > 0.6 ? 'bg-yellow-200' : 'bg-red-200'
                  }`}
                >
                  <span 
                    className={`absolute left-0 top-0 h-3 rounded-full ${
                      confidence > 0.75 ? 'bg-green-500' : 
                      confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                    }`} 
                    style={{ width: `${confidence * 100}%` }}
                  ></span>
                </span>
                <span className="ml-2">{(confidence * 100).toFixed(0)}%</span>
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Top 5 List */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <TrophyIcon />
                <span>Predicted Top 5</span>
              </h3>
              <div className="space-y-3">
                {predictionResults.top5.map((driver) => (
                  <div 
                    key={driver.position} 
                    className="flex items-center p-3 rounded-md border border-gray-200 hover:bg-gray-50 transition-colors"
                    style={{ borderLeftWidth: '4px', borderLeftColor: getDriverTeam(driver.driver_code) }}
                  >
                    <div className="flex-shrink-0 w-8 text-center font-bold text-lg">
                      {driver.position}
                    </div>
                    <div className="ml-3 flex-grow">
                      <div className="font-semibold">{driver.driver_code}</div>
                      <div className="text-sm text-gray-600">
                        Q3 Appearances: {driver.circuit_stats.q3_appearances}
                      </div>
                    </div>
                    <div className="flex-shrink-0 w-20 text-right font-medium">
                      {formatProbability(driver.probability)}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Probability Chart */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <CarIcon />
                <span>Probability Distribution</span>
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={createChartData()}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} />
                    <YAxis type="category" dataKey="name" />
                    <Tooltip formatter={(value) => [`${value}%`, "Probability"]} />
                    <Bar 
                      dataKey="probability" 
                      background={{ fill: '#eee' }}
                    >
                      {createChartData().map((entry, index) => (
                        <rect 
                          key={`rect-${index}`} 
                          fill={entry.fill} 
                          x={0} y={0} width={0} height={0} // These will be calculated by recharts
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Render the app
ReactDOM.render(
  <F1PredictionApp />,
  document.getElementById('root')
);