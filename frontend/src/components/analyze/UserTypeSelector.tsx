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
  const userType = useAppStore((s) => s.userType);
  const setUserType = useAppStore((s) => s.setUserType);

  return (
    <section className="space-y-3">
      <fieldset className="space-y-3">
        <legend className="block">
          <div className="text-lg font-semibold text-google-gray-900">User Type</div>
          <p className="text-sm text-google-gray-600">Choose the perspective for reporting (no sign-in required).</p>
        </legend>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3" role="radiogroup" aria-label="User type">
          {options.map(({ value, label, desc, icon: Icon, activeContainerClass, activeIconClass }) => {
            const active = userType === value;
            const id = `user-type-${value}`;

            return (
              <label
                key={value}
                htmlFor={id}
                className={[
                  'cursor-pointer text-left rounded-xl border p-4 transition-all shadow-sm hover:shadow-card',
                  'focus-within:ring-2 focus-within:ring-google-blue-500 focus-within:ring-offset-2',
                  active ? activeContainerClass : 'border-google-gray-200 bg-white hover:border-google-gray-300',
                ].join(' ')}
              >
                <input
                  id={id}
                  name="user-type"
                  type="radio"
                  value={value}
                  checked={active}
                  onChange={() => setUserType(value)}
                  className="sr-only"
                />

                <div className="flex items-start justify-between gap-3">
                  <div>
                    <Icon className={['w-6 h-6 mb-2', active ? activeIconClass : 'text-google-gray-600'].join(' ')} />
                    <div className="font-medium text-google-gray-900">{label}</div>
                    <div className="text-xs text-google-gray-600 mt-1">{desc}</div>
                  </div>

                  {active && (
                    <span className="inline-flex items-center rounded-full bg-white/70 px-2 py-0.5 text-xs font-medium text-google-gray-800 border border-google-gray-200">
                      Selected
                    </span>
                  )}
                </div>
              </label>
            );
          })}
        </div>
      </fieldset>
    </section>
  );
}
