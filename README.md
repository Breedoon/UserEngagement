# User Engagement Analysis
This repository contains the code for a small project on determining the most important factors influencing user engagement (i.e., adoption) for a given product using Logistic Regression and Random Forests.

### Synopsis of the Problem

The table `users` contains data on 12,000 users who signed up for the product in recent years, and looks like:  

| object_id | creation_time       | name              | email                      | creation_source | last_session_creation_time | opted_in_to_mailing_list | enabled_for_marketing_drip | org_id | invited_by_user_id | email_domain | 
|-----------|---------------------|-------------------|----------------------------|-----------------|----------------------------|--------------------------|----------------------------|--------|--------------------|--------------| 
| 1         | 2014-04-22 03:53:30 | Clausen August    | AugustCClausen@yahoo.com   | GUEST_INVITE    | 1398138810                 | 1                        | 0                          | 11     | 10803              | yahoo.com    | 
| 2         | 2013-11-15 03:45:04 | Poole Matthew     | MatthewPoole@gustr.com     | ORG_INVITE      | 1396237504                 | 0                        | 0                          | 1      | 316                | gustr.com    | 
| 3         | 2013-03-19 23:14:52 | Bottrill Mitchell | MitchellBottrill@gustr.com | ORG_INVITE      | 1363734892                 | 0                        | 0                          | 94     | 1525               | gustr.com    | 
| 4         | 2013-05-21 08:09:28 | Clausen Nicklas   | NicklasSClausen@yahoo.com  | GUEST_INVITE    | 1369210168                 | 0                        | 0                          | 1      | 5151               | yahoo.com    | 
| 5         | 2013-01-17 10:14:20 | Raw Grace         | GraceRaw@yahoo.com         | GUEST_INVITE    | 1358849660                 | 0                        | 0                          | 193    | 5240               | yahoo.com    | 

The table `user_engagement` has a row for each day that a user logged into the product, such as:

| time_stamp          | user_id | visited | 
|---------------------|---------|---------| 
| 2014-04-22 03:53:30 | 1       | 1       | 
| 2013-11-15 03:45:04 | 2       | 1       | 
| 2013-11-29 03:45:04 | 2       | 1       | 
| 2013-12-09 03:45:04 | 2       | 1       | 
  
Adopted users are defined as the users who logged into the product on three separate days in at least one seven-day period.

**Objective**: identify which factors influence adoption the most and can be leveraged to increase adoption. 

### Results
**Key findings**:
* People invited by an adopted user are 54% more likely to become adopted (20% rate compared to 13%).
* Only 7% of users invited to a personal project become adopted, and comprise just 10% of all adopted users.
* 57% of adopted users were invited to an organization (org or guest invite), and have higher adoption rates.
* The best predictor of adoption with an accuracy of ~90% is time used (i.e., people who have used the product the longest are most likely to be adopted), but is practically useless to increase adoption.
* The highest predictive accuracy that can be reasonably achieved without usage data is ~25%.

**Suggested Action to Increase Adoption**:
* Encourage current adopted users to invite people from their organization.
* Prioritize attracting new organizations rather than new individuals.
* Or, conversely, improve the product for personal use to fill in the gap in adoption.


![Venn diagrams of the data](dataviz/venns.png?raw=true "Overview of the data")  
_Figure 1. Overview of the data._



![Creation sources of adopted users](dataviz/adoption_sources.png?raw=true "Creation sources of adopted users")  
_Figure 2. Creation sources of adopted users._

![Influence of given factors on adoption](dataviz/influence.png?raw=true "Influence of given factors on adoption")  
_Figure 3. Influence of given factors on the rate of adoption (13% base rate) with a 95% CI._

### Appendix

_Table 1. Selected metrics for the relevant features._  


| Feature                    | Influence on base rate | Gini importance | Logit coefficient | Logit p-value | 
|----------------------------|------------------------|-----------------|-------------------|---------------| 
| inviter_adopted            | 1.54 &pm; 0.18             | 0.27            | 0.56              | 0.00          | 
| opted_in_to_mailing_list   | 1.04 &pm; 0.09             | 0.11            | 0.05              | 0.49          | 
| enabled_for_marketing_drip | 1.03 &pm; 0.12             | 0.11            | 0.01              | 0.95          | 
| invited_by_user            | 1.07 &pm; 0.06             | 0.02            | -1.25             | 1.00          | 
| PERSONAL PROJECTS          | 0.58 &pm; 0.09             | 0.26            | -2.49             | 0.00          | 
| GUEST_INVITE               | 1.25 &pm; 0.12             | 0.07            | -0.48             | 1.00          | 
| ORG_INVITE                 | 0.97 &pm; 0.08             | 0.06            | -0.77             | 1.00          | 
| SIGNUP                     | 1.05 &pm; 0.11             | 0.03            | -1.83             | 0.00          | 
| SIGNUP_GOOGLE_AUTH         | 1.25 &pm; 0.15             | 0.06            | -1.62             | 0.00          | 
