import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Plus,
  Search,
  Server,
  Cpu,
  Heart,
  Car,
  Rocket,
  Factory,
  Activity,
  ChevronRight,
  Loader2
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';
import type { System } from '../types';

const systemTypeIcons: Record<string, React.ElementType> = {
  vehicle: Car,
  robot: Cpu,
  medical_device: Heart,
  aerospace: Rocket,
  industrial: Factory,
  default: Server,
};

function getStatusColor(status: string) {
  switch (status) {
    case 'active': return 'bg-green-500';
    case 'anomaly_detected': return 'bg-orange-500';
    case 'maintenance': return 'bg-yellow-500';
    case 'inactive': return 'bg-slate-500';
    default: return 'bg-slate-500';
  }
}

function getHealthColor(score: number) {
  if (score >= 90) return 'text-green-500';
  if (score >= 70) return 'text-yellow-500';
  return 'text-red-500';
}

export default function Systems() {
  const navigate = useNavigate();
  const [systems, setSystems] = useState<System[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadSystems();
  }, []);

  const loadSystems = async () => {
    try {
      const data = await systemsApi.list();
      setSystems(data);
    } catch (error) {
      console.error('Failed to load systems:', error);
      // Use mock data if API fails
      setSystems([
        {
          id: '1',
          name: 'Fleet Vehicle Alpha',
          system_type: 'vehicle',
          status: 'anomaly_detected',
          health_score: 87.5,
          created_at: '2024-01-01T00:00:00Z',
        },
        {
          id: '2',
          name: 'Robot Arm Unit 7',
          system_type: 'robot',
          status: 'active',
          health_score: 94.2,
          created_at: '2024-01-02T00:00:00Z',
        },
        {
          id: '3',
          name: 'Medical Scanner MRI-3',
          system_type: 'medical_device',
          status: 'active',
          health_score: 99.1,
          created_at: '2024-01-03T00:00:00Z',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const filteredSystems = systems.filter(system =>
    system.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    system.system_type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Systems</h1>
          <p className="text-slate-400 mt-1">
            Manage and monitor your connected systems
          </p>
        </div>
        <button
          onClick={() => navigate('/systems/new')}
          className="flex items-center gap-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors"
        >
          <Plus className="w-5 h-5" />
          Add System
        </button>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
        <input
          type="text"
          placeholder="Search systems..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors"
        />
      </div>

      {/* Systems Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
        </div>
      ) : filteredSystems.length === 0 ? (
        <div className="text-center py-12">
          <Server className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400 mb-2">No systems found</p>
          <button
            onClick={() => navigate('/systems/new')}
            className="text-primary-400 hover:text-primary-300 font-medium"
          >
            Add your first system
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredSystems.map((system) => {
            const Icon = systemTypeIcons[system.system_type] || systemTypeIcons.default;
            return (
              <Link
                key={system.id}
                to={`/systems/${system.id}`}
                className="bg-slate-800 rounded-xl border border-slate-700 p-6 hover:border-primary-500/50 transition-colors group"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="p-3 bg-slate-700 rounded-lg group-hover:bg-primary-500/10 transition-colors">
                    <Icon className="w-6 h-6 text-slate-300 group-hover:text-primary-400 transition-colors" />
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={clsx('w-2 h-2 rounded-full', getStatusColor(system.status))} />
                    <span className="text-sm text-slate-400 capitalize">
                      {system.status.replace('_', ' ')}
                    </span>
                  </div>
                </div>

                <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-primary-400 transition-colors">
                  {system.name}
                </h3>
                <p className="text-sm text-slate-400 mb-4 capitalize">
                  {system.system_type.replace('_', ' ')}
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-slate-500" />
                    <span className={clsx('font-medium', getHealthColor(system.health_score))}>
                      {system.health_score}% Health
                    </span>
                  </div>
                  <ChevronRight className="w-5 h-5 text-slate-500 group-hover:text-primary-400 transition-colors" />
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
