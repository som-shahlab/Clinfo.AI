import requests

class SemanticScholarAPI:
    def __init__(self, timeout:float=20, offset:int=0, limit:int=30, fields:list=["url", "abstract"],retry_count:int=3):
        self.timeout     = timeout
        self.limit       = limit
        self.fields      = fields
        self.offset      = offset
        self.retry_count = retry_count

    def search(self, query , verbose=False):
        fields_  = ",".join(self.fields)
        query_   = query.replace(" ", "+")
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_}&offset={self.offset}&limit={self.limit}&fields={fields_}"
        headers = {"Accept": "application/json",'x-api-key': "a0Z1F23s6o3iGu3d8C3aA6cbEtn7Kr2naY5fWyDo"}

        try:
            response = requests.get(url, timeout=self.timeout, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                if verbose:
                    print(f"Error: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print("Timeout error: request timed out.")
            return "TIMEOUT"

        except requests.exceptions.HTTPError as errh:
            print("Http Error:", errh)
            return "HTTPERROR"

        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
            return "CONNECTIONERROR"

    def filter_papers(self, papers, threshold=5,minimum_return=5,maxmum_return=15,verbose=False):
        filtered_papers = []
        for paper in papers:
            if paper['abstract'] is None:
                # Skip papers with no abstract
                continue
            
            if paper['citationCount'] >= threshold:
                filtered_papers.append(paper)

        # If we don't have enough papers, lower the threshold
        if len(filtered_papers) < minimum_return:
            print("Lowering threshold to 1")
            filtered_papers = self.filter_papers(papers, threshold=1,minimum_return=-1,maxmum_return=maxmum_return,verbose=verbose)
            return filtered_papers

        if verbose:
            print(f"Filtered {len(papers)} papers down to {len(filtered_papers)} influential papers with at least {threshold} influential citations.")
        return filtered_papers[0:maxmum_return]
    
    def search_with_retry(self, query, verbose=False):
        attempt = 1
        while attempt <= self.retry_count:
            result = self.search(query=query,verbose=verbose)
            if result == "TIMEOUT":
                print(f"Retry attempt {attempt} of {self.retry_count}")
                attempt += 1
            else:
                return result

        print("Max retry count reached. Giving up.")
        return None

    def search_with_filter( self, query, threshold=1,minimum_return=5,maxmum_return=15,verbose=False):
        result = self.search_with_retry(query=query,verbose=verbose)
        if result is None:
            return None
        else:
            return self.filter_papers(result["data"],verbose=verbose,minimum_return=minimum_return,maxmum_return=maxmum_return,threshold=threshold)