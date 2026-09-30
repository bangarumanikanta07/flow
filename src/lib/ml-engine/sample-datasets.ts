import { DatasetSummary, DatasetColumnInfo } from './types';

// Deterministic Pseudo-random number generator for reproducible sample generation
class SeededRandom {
  private seed: number;
  constructor(seed: number = 42) {
    this.seed = seed % 2147483647;
    if (this.seed <= 0) this.seed += 2147483646;
  }
  next(): number {
    this.seed = (this.seed * 16807) % 2147483647;
    return (this.seed - 1) / 2147483646;
  }
  gaussian(mean = 0, stdev = 1): number {
    const u1 = 1.0 - this.next();
    const u2 = 1.0 - this.next();
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    return mean + stdev * randStdNormal;
  }
}

export function generateCustomerChurnDataset(): { rawCsv: string; summary: DatasetSummary } {
  const rng = new SeededRandom(101);
  const rows: Record<string, string | number>[] = [];
  const count = 500;

  for (let i = 0; i < count; i++) {
    const tenureMonths = Math.max(1, Math.round(rng.gaussian(28, 18)));
    const contract = tenureMonths > 24 ? (rng.next() > 0.4 ? 'Two Year' : 'One Year') : (rng.next() > 0.8 ? 'One Year' : 'Month-to-Month');
    const monthlyCharges = Math.round((45 + rng.gaussian(25, 15) + (contract === 'Month-to-Month' ? 10 : 0)) * 100) / 100;
    const boundedMonthly = Math.max(18.5, Math.min(118.5, monthlyCharges));
    const totalCharges = Math.round((boundedMonthly * tenureMonths + rng.gaussian(0, 50)) * 100) / 100;
    const supportTickets = Math.max(0, Math.round(rng.gaussian(contract === 'Month-to-Month' ? 3.2 : 1.1, 1.6)));
    const techSupport = rng.next() > 0.5 ? 'Yes' : 'No';
    const paperlessBilling = rng.next() > 0.35 ? 'Yes' : 'No';

    // Latent churn probability
    const z = -0.045 * tenureMonths + 0.028 * boundedMonthly + 0.45 * supportTickets + (contract === 'Month-to-Month' ? 0.9 : -1.2) + (techSupport === 'No' ? 0.4 : -0.3) - 1.2;
    const prob = 1 / (1 + Math.exp(-z));
    const churn = rng.next() < prob ? 1 : 0;

    rows.push({
      customer_id: `CUST-${1000 + i}`,
      tenure_months: tenureMonths,
      monthly_charges: boundedMonthly,
      total_charges: Math.max(boundedMonthly, totalCharges),
      support_tickets: supportTickets,
      contract_type: contract,
      tech_support: techSupport,
      paperless_billing: paperlessBilling,
      churn: churn
    });
  }

  const columns = buildColumnSummaries(rows, 'churn');
  const headers = Object.keys(rows[0]);
  const rawCsv = [
    headers.join(','),
    ...rows.map(r => headers.map(h => r[h]).join(','))
  ].join('\n');

  return {
    rawCsv,
    summary: {
      id: 'churn_v1',
      name: 'Telco Customer Churn & Retention',
      rowCount: rows.length,
      columnCount: headers.length,
      columns,
      targetColumn: 'churn',
      taskType: 'classification',
      features: headers.filter(h => h !== 'churn' && h !== 'customer_id'),
      headRows: rows.slice(0, 15),
      missingValuesTotal: 0
    }
  };
}

export function generateCreditRiskDataset(): { rawCsv: string; summary: DatasetSummary } {
  const rng = new SeededRandom(202);
  const rows: Record<string, string | number>[] = [];
  const count = 450;

  for (let i = 0; i < count; i++) {
    const annualIncome = Math.round(Math.max(24000, rng.gaussian(68000, 22000)));
    const debtToIncome = Math.round(Math.max(4, Math.min(58, rng.gaussian(24, 11))) * 10) / 10;
    const creditScore = Math.round(Math.max(480, Math.min(840, rng.gaussian(680, 75))));
    const openCreditLines = Math.max(1, Math.round(rng.gaussian(8, 3.5)));
    const loanAmount = Math.round(Math.max(3000, rng.gaussian(18000, 7500)));
    const employmentYears = Math.max(0, Math.round(rng.gaussian(6, 4)));

    // Risk calculation
    const riskScore = -0.00004 * annualIncome + 0.07 * debtToIncome - 0.016 * creditScore + 0.00006 * loanAmount - 0.15 * employmentYears + 8.5;
    const prob = 1 / (1 + Math.exp(-riskScore));
    const isDefault = rng.next() < prob ? 1 : 0;

    rows.push({
      annual_income: annualIncome,
      debt_to_income: debtToIncome,
      credit_score: creditScore,
      open_credit_lines: openCreditLines,
      loan_amount: loanAmount,
      employment_years: employmentYears,
      is_default: isDefault
    });
  }

  const columns = buildColumnSummaries(rows, 'is_default');
  const headers = Object.keys(rows[0]);
  const rawCsv = [
    headers.join(','),
    ...rows.map(r => headers.map(h => r[h]).join(','))
  ].join('\n');

  return {
    rawCsv,
    summary: {
      id: 'credit_risk_v1',
      name: 'Consumer Credit Default Assessment',
      rowCount: rows.length,
      columnCount: headers.length,
      columns,
      targetColumn: 'is_default',
      taskType: 'classification',
      features: headers.filter(h => h !== 'is_default'),
      headRows: rows.slice(0, 15),
      missingValuesTotal: 0
    }
  };
}

export function generateHousingPricingDataset(): { rawCsv: string; summary: DatasetSummary } {
  const rng = new SeededRandom(303);
  const rows: Record<string, string | number>[] = [];
  const count = 400;

  for (let i = 0; i < count; i++) {
    const sqft = Math.round(Math.max(650, rng.gaussian(1950, 550)));
    const bedrooms = Math.max(1, Math.min(6, Math.round(sqft / 550 + rng.gaussian(0, 0.6))));
    const bathrooms = Math.max(1, Math.min(5, Math.round((bedrooms * 0.75 + rng.gaussian(0.5, 0.4)) * 2) / 2));
    const ageYears = Math.max(0, Math.round(rng.gaussian(18, 14)));
    const schoolRating = Math.max(1, Math.min(10, Math.round(rng.gaussian(6.8, 2))));
    const garageSpaces = Math.max(0, Math.min(3, Math.round(rng.gaussian(1.5, 0.8))));

    // Target: price tier (High Value = 1, Standard Value = 0)
    const baseValue = sqft * 185 - ageYears * 1200 + schoolRating * 18000 + bathrooms * 14000;
    const isHighTier = baseValue > 420000 ? 1 : 0;

    rows.push({
      sqft,
      bedrooms,
      bathrooms,
      age_years: ageYears,
      school_rating: schoolRating,
      garage_spaces: garageSpaces,
      high_value_tier: isHighTier
    });
  }

  const columns = buildColumnSummaries(rows, 'high_value_tier');
  const headers = Object.keys(rows[0]);
  const rawCsv = [
    headers.join(','),
    ...rows.map(r => headers.map(h => r[h]).join(','))
  ].join('\n');

  return {
    rawCsv,
    summary: {
      id: 'housing_v1',
      name: 'Residential Property Valuation',
      rowCount: rows.length,
      columnCount: headers.length,
      columns,
      targetColumn: 'high_value_tier',
      taskType: 'classification',
      features: headers.filter(h => h !== 'high_value_tier'),
      headRows: rows.slice(0, 15),
      missingValuesTotal: 0
    }
  };
}

export function generateCreditCardFraudDataset(): { rawCsv: string; summary: DatasetSummary } {
  const rng = new SeededRandom(707);
  const rows: Record<string, string | number>[] = [];
  const count = 1000;
  const merchantCategories = ['grocery', 'electronics', 'travel', 'dining', 'online_retail', 'gambling'];

  for (let i = 0; i < count; i++) {
    const transactionAmount = Math.round(Math.max(1.5, rng.gaussian(115, 110)) * 100) / 100;
    const transactionHour = Math.max(0, Math.min(23, Math.round(rng.gaussian(14, 5))));
    const transactionsLast24h = Math.max(1, Math.round(rng.gaussian(4.5, 3.2)));
    const accountAgeDays = Math.max(1, Math.round(rng.gaussian(420, 260)));
    const distanceFromHomeKm = Math.round(Math.max(0.1, rng.gaussian(18, 30)) * 10) / 10;
    const deviceRiskScore = Math.round(Math.max(0.01, Math.min(0.99, rng.gaussian(0.28, 0.24))) * 100) / 100;
    const failedLoginAttempts = Math.max(0, Math.round(rng.gaussian(0.4, 0.9)));
    const merchantCategory = merchantCategories[Math.floor(rng.next() * merchantCategories.length)];

    // Realistic non-linear latent fraud probability
    const isLateNight = transactionHour < 5 || transactionHour > 23 ? 1.0 : -0.4;
    const isHighRiskCategory = merchantCategory === 'gambling' || merchantCategory === 'travel' ? 1.2 : -0.3;
    const isNewAccount = accountAgeDays < 30 ? 1.3 : -0.5;

    const z =
      0.005 * transactionAmount +
      0.03 * distanceFromHomeKm +
      0.22 * transactionsLast24h +
      0.8 * failedLoginAttempts +
      2.8 * deviceRiskScore +
      isLateNight +
      isHighRiskCategory +
      isNewAccount -
      4.8;

    const prob = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, z))));
    const isFraud = rng.next() < prob ? 1 : 0;

    rows.push({
      transaction_amount: transactionAmount,
      transaction_hour: transactionHour,
      transactions_last_24h: transactionsLast24h,
      account_age_days: accountAgeDays,
      distance_from_home_km: distanceFromHomeKm,
      device_risk_score: deviceRiskScore,
      failed_login_attempts: failedLoginAttempts,
      merchant_category: merchantCategory,
      is_fraud: isFraud
    });
  }

  const columns = buildColumnSummaries(rows, 'is_fraud');
  const headers = Object.keys(rows[0]);
  const rawCsv = [
    headers.join(','),
    ...rows.map(r => headers.map(h => r[h]).join(','))
  ].join('\n');

  return {
    rawCsv,
    summary: {
      id: 'fraud_v1',
      name: 'Credit Card Fraud Detection (1,000 Transactions)',
      rowCount: rows.length,
      columnCount: headers.length,
      columns,
      targetColumn: 'is_fraud',
      taskType: 'classification',
      features: headers.filter(h => h !== 'is_fraud'),
      headRows: rows.slice(0, 15),
      missingValuesTotal: 0
    }
  };
}

export function buildColumnSummaries(
  rows: Record<string, string | number>[],
  targetColumn: string
): DatasetColumnInfo[] {
  if (rows.length === 0) return [];
  const headers = Object.keys(rows[0]);

  return headers.map(header => {
    const values = rows.map(r => r[header]);
    const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
    const isNumeric = nonNullValues.every(v => typeof v === 'number' || (!isNaN(Number(v)) && typeof v === 'string'));
    const uniqueValues = Array.from(new Set(values));

    if (isNumeric) {
      const nums = nonNullValues.map(v => Number(v));
      const mean = nums.reduce((a, b) => a + b, 0) / (nums.length || 1);
      const variance = nums.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (nums.length || 1);
      const std = Math.sqrt(variance);
      const min = Math.min(...nums);
      const max = Math.max(...nums);

      return {
        name: header,
        type: 'numerical',
        nonNullCount: nonNullValues.length,
        nullCount: values.length - nonNullValues.length,
        uniqueCount: uniqueValues.length,
        mean: Math.round(mean * 1000) / 1000,
        std: Math.round(std * 1000) / 1000,
        min: Math.round(min * 1000) / 1000,
        max: Math.round(max * 1000) / 1000,
        sampleValues: uniqueValues.slice(0, 5)
      };
    } else {
      return {
        name: header,
        type: 'categorical',
        nonNullCount: nonNullValues.length,
        nullCount: values.length - nonNullValues.length,
        uniqueCount: uniqueValues.length,
        sampleValues: uniqueValues.slice(0, 5)
      };
    }
  });
}
