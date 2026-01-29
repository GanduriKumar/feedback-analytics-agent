import { Briefcase, Wrench, Headphones, LineChart, Crown } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import type { UserType } from '../../types';

const options: Array<{
  value: UserType;
  label: string;
  desc: string;
  icon: any;
  activeContainerClass: string;
  activeIconClass: string;
}> = [
  {
    value: 'product-manager',
    label: 'Product Manager',
    desc: 'Priorities, roadmap, customer needs',
    icon: Briefcase,
    activeContainerClass: 'border-google-blue-300 bg-google-blue-50',
    activeIconClass: 'text-google-blue-700',
  },
  {
    value: 'engineer',
    label: 'Engineer',
    desc: 'Bugs, performance, technical themes',
    icon: Wrench,
    activeContainerClass: 'border-google-green-300 bg-google-green-50',
    activeIconClass: 'text-google-green-700',
  },
  {
    value: 'support',
    label: 'Support',
    desc: 'Top pain points & recurring complaints',
    icon: Headphones,
    activeContainerClass: 'border-google-yellow-300 bg-google-yellow-50',
    activeIconClass: 'text-google-yellow-700',
  },
  {
    value: 'business-analyst',
    label: 'Business Analyst',
    desc: 'Trends, metrics, distribution insights',
    icon: LineChart,
    activeContainerClass: 'border-google-red-300 bg-google-red-50',
    activeIconClass: 'text-google-red-700',
  },
  {
    value: 'executive',
    label: 'Executive',
    desc: 'High-level snapshot & recommendations',
    icon: Crown,
    activeContainerClass: 'border-google-gray-300 bg-google-gray-100',
    activeIconClass: 'text-google-gray-800',
  },
];

export function UserTypeSelector() {
  const { userType, setUserType } = useAppStore();

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold text-google-gray-900">User Type</h2>
        <p className="text-sm text-google-gray-600">Choose the perspective for reporting (no sign-in required).</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
        {options.map(({ value, label, desc, icon: Icon, activeContainerClass, activeIconClass }) => {
          const active = userType === value;
          return (
            <button
              key={value}
              type="button"
              onClick={() => setUserType(value)}
              className={[
                'text-left rounded-xl border p-4 transition-all shadow-sm hover:shadow-card',
                active ? activeContainerClass : 'border-google-gray-200 bg-white hover:border-google-gray-300',
              ].join(' ')}
            >
              <Icon className={['w-6 h-6 mb-2', active ? activeIconClass : 'text-google-gray-600'].join(' ')} />
              <div className="font-medium text-google-gray-900">{label}</div>
              <div className="text-xs text-google-gray-600 mt-1">{desc}</div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
