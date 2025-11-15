# data_manager.py (FINAL, v35 - Headerless SDN Fix)

import pandas as pd
from fuzzywuzzy import process 
import os
import streamlit as st # Used for error display

class DataManager:
    """
    Manages all data loading and querying for the AML application.
    Loads data once upon initialization.
    """
    
    def __init__(self, accounts_file='accounts.csv', transactions_file='transactions.csv', sanctions_file='sdn.csv'):
        print("Initializing DataManager...")
        try:
            # Store file paths
            self.accounts_file = accounts_file
            self.transactions_file = transactions_file
            self.sanctions_file = sanctions_file
            self.closed_cases_file = 'data/closed_cases.csv' # Archive file
            
            # Load all data into memory
            self.accounts_df = pd.read_csv(self.accounts_file)
            self.transactions_df = pd.read_csv(self.transactions_file)
            
            # --- ⬇️ FIX FOR HEADERLESS SDN.CSV ⬇️ ---
            try:
                # Assign names. Based on your screenshot, the name is in the second column (index 1).
                col_names = ['ID', 'NAME', 'Type', 'Country', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12']
                self.sanctions_df = pd.read_csv(
                    self.sanctions_file, 
                    header=None, # Tell pandas there is no header
                    names=col_names, # Assign our own names
                    usecols=['ID', 'NAME'], # We only really need these two
                    on_bad_lines='skip',
                    encoding='latin1' # Try a different encoding if utf-8 fails
                )
                print("DataManager: Loaded sdn.csv with custom headers. 'NAME' column assigned to column 2.")
            except Exception as e:
                print(f"ERROR: Failed to load headerless sdn.csv: {e}")
                # Load an empty dataframe with the expected 'NAME' column to avoid downstream errors
                self.sanctions_df = pd.DataFrame(columns=['NAME'])
            # --- ⬆️ END OF FIX ⬆️ ---

            # --- Data Sanitization ---
            self.accounts_df['ACCOUNT_ID'] = self.accounts_df['ACCOUNT_ID'].astype(str)
            self.transactions_df['SENDER_ACCOUNT_ID'] = self.transactions_df['SENDER_ACCOUNT_ID'].astype(str)
            self.transactions_df['RECEIVER_ACCOUNT_ID'] = self.transactions_df['RECEIVER_ACCOUNT_ID'].astype(str)
            
            
            # --- FIX FOR 'accounts.csv' (Resolves "No Report" Bug) ---
            account_name_col_found = False
            common_account_names = ['NAME', 'name', 'full_name', 'customer_name', 'CUSTOMER_NAME'] 
            
            for col_name in common_account_names:
                if col_name in self.accounts_df.columns:
                    self.accounts_df.rename(columns={col_name: 'NAME'}, inplace=True)
                    print(f"DataManager: Found and renamed account name column '{col_name}' to 'NAME'.")
                    account_name_col_found = True
                    break
            
            if not account_name_col_found:
                print("WARNING: Could not find a name column in accounts.csv. Sanctions checks may use Account ID.")
            
            # --- Prepare Sanctions List (Now that 'NAME' column is guaranteed) ---
            if 'NAME' in self.sanctions_df.columns:
                self.sanctions_names = self.sanctions_df['NAME'].dropna().tolist()
            else:
                print("ERROR: 'NAME' column still not found in sanctions_df even after custom loading.")
                self.sanctions_names = []

            print("DataManager: All data loaded and SANITIZED successfully.")

        except FileNotFoundError as e:
            print(f"FATAL ERROR: Could not find data file: {e.filename}")
            st.error(f"FATAL ERROR: Could not find data file: {e.filename}. Make sure it is at '{e.filename}'")
            raise
        except Exception as e:
            print(f"FATAL ERROR: Failed to load data: {e}")
            st.error(f"FATAL ERROR: Failed to load data: {e}")
            raise

    # ... (all other functions: get_suspicious_alerts, get_account_details, etc. remain unchanged) ...

    def get_suspicious_alerts(self, num_alerts=None):
        """
        Gets a list of suspicious accounts (IS_FRAUD == True).
        """
        try:
            suspicious_accounts = self.accounts_df[
                self.accounts_df['IS_FRAUD'] == True
            ]
            account_ids = suspicious_accounts['ACCOUNT_ID'].tolist()
            print(f"DataManager: Found {len(account_ids)} suspicious accounts.")
            
            if num_alerts:
                return account_ids[:num_alerts]
            else:
                return account_ids

        except KeyError:
            print("ERROR: 'IS_FRAUD' column not found in accounts.csv. Returning empty list.")
            return []

    def get_account_details(self, customer_id: str) -> dict:
        """
        Fetches the KYC details for a single account.
        """
        try:
            customer_id = str(customer_id) 
            account_details = self.accounts_df[
                self.accounts_df['ACCOUNT_ID'] == customer_id
            ]
            
            if not account_details.empty:
                return account_details.iloc[0].to_dict()
            else:
                return {"Error": f"Account ID {customer_id} not found."}
        except Exception as e:
            return {"Error": f"Failed to get account details: {e}"}

    def get_transaction_cluster(self, customer_id: str) -> pd.DataFrame:
        """
        Fetches all transactions (inbound and outbound) for a single account.
        """
        try:
            customer_id = str(customer_id)
            
            inbound = self.transactions_df[
                self.transactions_df['RECEIVER_ACCOUNT_ID'] == customer_id
            ]
            outbound = self.transactions_df[
                self.transactions_df['SENDER_ACCOUNT_ID'] == customer_id
            ]
            
            cluster_df = pd.concat([inbound, outbound]).drop_duplicates()
            
            if 'TIMESTAMP' in cluster_df.columns:
                cluster_df = cluster_df.sort_values(by='TIMESTAMP')
                
            return cluster_df

        except Exception as e:
            print(f"Error getting transaction cluster: {e}")
            return pd.DataFrame() 

    def check_sanctions_list(self, customer_name: str) -> dict:
        """
        Checks a customer's name against the loaded sanctions list using
        fuzzy matching.
        """
        if not self.sanctions_names:
            return {"Error": "Sanctions list is not loaded."}
            
        try:
            # Check if customer_name is just a number (like an ID)
            if str(customer_name).isnumeric():
                return {"hit": False, "note": f"Search was not performed. '{customer_name}' is an ID, not a name."}
                
            best_match = process.extractOne(customer_name, self.sanctions_names)
            
            if best_match and best_match[1] > 90:
                return {
                    "hit": True,
                    "match_name": best_match[0],
                    "score": best_match[1]
                }
            elif best_match and best_match[1] > 65: 
                return {
                    "hit": False,
                    "note": f"No direct hit, but a score of {best_match[1]} was found for '{best_match[0]}'. This could be a partial match.",
                    "match_name": best_match[0],
                    "score": best_match[1]
                }
            else:
                return {
                    "hit": False,
                    "match_name": best_match[0] if best_match else None,
                    "score": best_match[1] if best_match else 0
                }
        
        except Exception as e:
            return {"Error": f"Failed during sanctions check: {e}"}

    def archive_case(self, case_id: str, decision: str, brief: str):
        """
        Saves a completed investigation to the 'closed_cases.csv' file.
        """
        try:
            new_case_df = pd.DataFrame({
                'case_id': [case_id],
                'decision': [decision],
                'brief': [brief],
                'timestamp': [pd.Timestamp.now()] 
            })
            
            if not os.path.exists(self.closed_cases_file):
                new_case_df.to_csv(self.closed_cases_file, index=False)
            else:
                new_case_df.to_csv(self.closed_cases_file, mode='a', header=False, index=False)
            
            print(f"DataManager: Successfully archived case {case_id}")
            return True

        except Exception as e:
            print(f"ERROR: Failed to archive case {case_id}: {e}")
            return False

    def get_closed_cases(self) -> pd.DataFrame:
        """
        Reads and returns all completed investigations from 'closed_cases.csv'.
        """
        if not os.path.exists(self.closed_cases_file):
            return pd.DataFrame(columns=['case_id', 'decision', 'brief', 'timestamp'])
            
        try:
            df = pd.read_csv(self.closed_cases_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values(by='timestamp', ascending=False)
        
        except Exception as e:
            print(f"ERROR: Failed to get closed cases: {e}")
            return pd.DataFrame(columns=['case_id', 'decision', 'brief', 'timestamp'])