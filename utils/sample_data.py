import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SampleDataGenerator:
    """Generates sample datasets for demonstration purposes"""
    
    def __init__(self):
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def generate_sales_data(self, n_records=1000):
        """Generate sample sales dataset"""
        
        # Date range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = pd.date_range(start_date, end_date).tolist()
        
        # Generate data
        data = []
        
        products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smart Watch', 'Monitor', 'Keyboard', 'Mouse']
        regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa']
        categories = ['Electronics', 'Accessories', 'Computing', 'Mobile']
        sales_channels = ['Online', 'Retail Store', 'Partner', 'Direct Sales']
        
        for i in range(n_records):
            # Base price varies by product
            product = random.choice(products)
            base_prices = {
                'Laptop': 1200, 'Smartphone': 800, 'Tablet': 500, 
                'Headphones': 150, 'Smart Watch': 300, 'Monitor': 400,
                'Keyboard': 100, 'Mouse': 50
            }
            
            base_price = base_prices.get(product, 200)
            price = base_price * (0.8 + random.random() * 0.4)  # ±20% variation
            
            quantity = max(1, int(np.random.poisson(3)))  # Poisson distribution for quantity
            revenue = price * quantity
            
            # Add seasonal effects
            date = random.choice(date_range)
            if date.month in [11, 12]:  # Holiday season boost
                revenue *= 1.2
            
            record = {
                'Date': date,
                'Product': product,
                'Category': random.choice(categories),
                'Region': random.choice(regions),
                'Sales_Channel': random.choice(sales_channels),
                'Quantity': quantity,
                'Unit_Price': round(price, 2),
                'Revenue': round(revenue, 2),
                'Customer_Satisfaction': round(3.5 + random.random() * 1.5, 1),  # 3.5-5.0 scale
                'Marketing_Spend': round(revenue * (0.05 + random.random() * 0.15), 2)  # 5-20% of revenue
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        return df.sort_values('Date').reset_index(drop=True)
    
    def generate_employee_data(self, n_records=500):
        """Generate sample employee dataset"""
        
        departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations', 'Customer Support']
        positions = {
            'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Engineering Manager'],
            'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Content Creator', 'Digital Marketer'],
            'Sales': ['Sales Rep', 'Senior Sales Rep', 'Sales Manager', 'Account Executive'],
            'HR': ['HR Generalist', 'HR Manager', 'Recruiter', 'HR Director'],
            'Finance': ['Financial Analyst', 'Senior Analyst', 'Finance Manager', 'Controller'],
            'Operations': ['Operations Specialist', 'Operations Manager', 'Project Manager'],
            'Customer Support': ['Support Specialist', 'Senior Support', 'Support Manager']
        }
        
        first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward', 'Fiona', 'George', 'Hannah', 
                      'Ian', 'Julia', 'Kevin', 'Linda', 'Michael', 'Nina', 'Oliver', 'Patricia']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 
                     'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson']
        
        data = []
        
        for i in range(n_records):
            department = random.choice(departments)
            position = random.choice(positions[department])
            
            # Salary based on department and seniority
            base_salaries = {
                'Engineering': 85000, 'Marketing': 65000, 'Sales': 70000,
                'HR': 60000, 'Finance': 75000, 'Operations': 68000,
                'Customer Support': 55000
            }
            
            base_salary = base_salaries[department]
            if 'Senior' in position or 'Lead' in position:
                base_salary *= 1.3
            elif 'Manager' in position or 'Director' in position:
                base_salary *= 1.6
            
            salary = int(base_salary * (0.8 + random.random() * 0.4))
            
            # Years of experience
            experience = max(0, int(np.random.gamma(3, 2)))  # Gamma distribution for experience
            
            # Performance rating
            performance = round(2.5 + random.random() * 2.5, 1)  # 2.5-5.0 scale
            
            # Hire date
            hire_date = datetime.now() - timedelta(days=random.randint(30, 2000))
            
            record = {
                'Employee_ID': f'EMP{i+1:04d}',
                'First_Name': random.choice(first_names),
                'Last_Name': random.choice(last_names),
                'Department': department,
                'Position': position,
                'Hire_Date': hire_date.date(),
                'Years_Experience': experience,
                'Salary': salary,
                'Performance_Rating': performance,
                'Training_Hours': random.randint(10, 120),
                'Remote_Work_Days': random.randint(0, 5),
                'Bonus_Percentage': round(random.random() * 20, 1)
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_customer_data(self, n_records=800):
        """Generate sample customer dataset"""
        
        segments = ['Premium', 'Standard', 'Basic']
        acquisition_channels = ['Social Media', 'Search Engine', 'Email', 'Referral', 'Direct', 'Advertising']
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'Brazil']
        
        data = []
        
        for i in range(n_records):
            # Customer lifecycle metrics
            acquisition_date = datetime.now() - timedelta(days=random.randint(1, 1095))  # Up to 3 years
            days_since_acquisition = (datetime.now() - acquisition_date).days
            
            # Segment affects various metrics
            segment = random.choice(segments)
            segment_multipliers = {'Premium': 3.0, 'Standard': 1.5, 'Basic': 1.0}
            multiplier = segment_multipliers[segment]
            
            # Lifetime value
            ltv = int(random.uniform(100, 2000) * multiplier)
            
            # Purchase frequency
            avg_days_between_purchases = int(30 / multiplier + random.uniform(-10, 10))
            avg_days_between_purchases = max(1, avg_days_between_purchases)
            
            # Satisfaction and engagement
            satisfaction = round(3.0 + random.random() * 2.0, 1)  # 3.0-5.0 scale
            engagement_score = round(random.uniform(0.2, 1.0) * multiplier, 2)
            
            record = {
                'Customer_ID': f'CUST{i+1:05d}',
                'Acquisition_Date': acquisition_date.date(),
                'Days_Since_Acquisition': days_since_acquisition,
                'Segment': segment,
                'Country': random.choice(countries),
                'Acquisition_Channel': random.choice(acquisition_channels),
                'Lifetime_Value': ltv,
                'Total_Orders': max(1, int(days_since_acquisition / avg_days_between_purchases)),
                'Avg_Order_Value': round(ltv / max(1, int(days_since_acquisition / avg_days_between_purchases)), 2),
                'Days_Since_Last_Purchase': random.randint(0, avg_days_between_purchases * 2),
                'Customer_Satisfaction': satisfaction,
                'Engagement_Score': engagement_score,
                'Support_Tickets': max(0, int(np.random.poisson(2) * (1 / multiplier))),
                'Newsletter_Subscriber': random.choice([True, False]),
                'Churn_Risk': round(random.uniform(0.1, 0.9), 2)
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_financial_data(self, n_records=600):
        """Generate sample financial dataset"""
        
        categories = ['Revenue', 'Cost of Goods Sold', 'Marketing', 'Sales', 'R&D', 
                     'Operations', 'Administrative', 'Other Income', 'Taxes']
        
        subcategories = {
            'Revenue': ['Product Sales', 'Service Revenue', 'Subscription Revenue', 'Licensing'],
            'Cost of Goods Sold': ['Materials', 'Labor', 'Manufacturing Overhead'],
            'Marketing': ['Digital Advertising', 'Traditional Advertising', 'Events', 'Content Creation'],
            'Sales': ['Salaries', 'Commissions', 'Travel', 'Tools and Software'],
            'R&D': ['Personnel', 'Equipment', 'Patents', 'External Research'],
            'Operations': ['Facilities', 'Utilities', 'Equipment Maintenance', 'Insurance'],
            'Administrative': ['Executive Salaries', 'Legal', 'Accounting', 'Office Supplies'],
            'Other Income': ['Investment Income', 'Asset Sales', 'Foreign Exchange'],
            'Taxes': ['Corporate Tax', 'Sales Tax', 'Property Tax']
        }
        
        # Generate monthly data for 2 years
        start_date = datetime(2023, 1, 1)
        months = pd.date_range(start_date, periods=24, freq='MS')
        
        data = []
        
        for month in months:
            for category in categories:
                for subcategory in subcategories[category]:
                    # Base amounts vary by category and subcategory
                    base_amounts = {
                        'Revenue': (50000, 200000),
                        'Cost of Goods Sold': (20000, 80000),
                        'Marketing': (5000, 25000),
                        'Sales': (8000, 35000),
                        'R&D': (10000, 40000),
                        'Operations': (5000, 20000),
                        'Administrative': (3000, 15000),
                        'Other Income': (0, 5000),
                        'Taxes': (2000, 15000)
                    }
                    
                    min_amount, max_amount = base_amounts[category]
                    amount = random.uniform(min_amount, max_amount)
                    
                    # Add seasonal variation
                    if month.month in [11, 12] and category == 'Revenue':
                        amount *= 1.3  # Holiday boost
                    elif month.month in [1, 2] and category == 'Revenue':
                        amount *= 0.8  # Post-holiday drop
                    
                    # Make expenses negative
                    if category != 'Revenue' and category != 'Other Income':
                        amount = -amount
                    
                    record = {
                        'Date': month.date(),
                        'Category': category,
                        'Subcategory': subcategory,
                        'Amount': round(amount, 2),
                        'Budget': round(amount * random.uniform(0.9, 1.1), 2),  # Budget vs actual
                        'Variance': 0,  # Will calculate after
                        'Department': random.choice(['Corporate', 'Product', 'Sales', 'Marketing', 'Operations']),
                        'Cost_Center': f'CC{random.randint(1000, 9999)}',
                        'Approved_By': random.choice(['John Smith', 'Sarah Johnson', 'Mike Davis', 'Lisa Brown'])
                    }
                    
                    # Calculate variance
                    record['Variance'] = round(record['Amount'] - record['Budget'], 2)
                    
                    data.append(record)
        
        return pd.DataFrame(data)
    
    def get_available_datasets(self):
        """Return list of available sample datasets"""
        return {
            'Sales Performance Data': 'Comprehensive sales data with products, regions, and performance metrics',
            'Employee Analytics Data': 'HR dataset with employee information, performance, and compensation',
            'Customer Behavior Data': 'Customer lifecycle data with segments, satisfaction, and churn metrics',
            'Financial Performance Data': 'Financial data with revenue, expenses, and budget analysis'
        }