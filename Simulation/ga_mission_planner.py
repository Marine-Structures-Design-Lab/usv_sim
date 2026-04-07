'''
Genetic algorithm for optimal vessel mission planning

Authors: Rachel Mecca, Matt Collette

(c) 2024 Regents of the University of Michigan

'''
from random import Random
import numpy as np
import copy
from datetime import timedelta
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
import logging
entry = {'mpi_imported':False} 
globals().update(entry)
geod = Geodesic.WGS84
logger = logging.getLogger(__name__)
class VesselProblem():
    '''
    GA vessel mission planning problem class
    ----------------------------------------
    '''
    def __init__(self,vessel, start_time,speeds=[]):
        '''
        VesselProblem Constructor
        --------------------

        Parameters
        ---------------------
        :param vessel: vessel object
        :param start_time: datetime object, the start time of mission planning
        :param speeds: list of floats, speeds to choose from in mission planning
        '''
        self.vessel=vessel
        self.start_time=start_time
        self.start=vessel.start
        self.end=vessel.end
        self.waypoints=[self.start]
        self.speeds=speeds
        if(self.speeds==[]):
           for i in range(1,5):
              self.speeds.append(vessel.speed/i)
        for dest in self.vessel.full_mission.destinations:
           if(not dest.visited):
              self.waypoints.append(dest)
        if(self.vessel.opt_mission!=None):
         for dest in self.vessel.opt_mission.destinations:
           if(dest.visited):
              self.start=dest
              self.waypoints[0]=dest
        self.waypoints.append(self.end)
        self.max_priority=vessel.full_mission.total_priority        
        
class Individual:
    '''
    Stores identities of each chromosome
    '''
    def __init__(self,chromosome,generation,violation_sum,total_priority,fuel_left,fuel_used):
        '''
        Individual Constructor
        --------------------------------------------------------------------------
        
        Parameters
        ---------------------------------------------------------------------------
        :param chromosome: 2D list with 2 equal sized rows
        :param first row indicates the order of destination numbers to be visited(ints)
        :param second row indicates the vessel's speed to get to each destination(floats)
        :param generation: int, the generation number the individual is part of
        :param violation_sum: float, the amount of constraint violation, 0 indicates no violation
        :param total_priority: int, normalized total priority of the mission represented by the chromosome
        :param fuel_left: float, normalized amount of fuel remaing after completing the mission
        :param fuel_used: float, total fuel used in kN by the mission represented by the chromosome 
       '''
        self.chromosome=chromosome
        self.generation=generation
        self.violation_sum = violation_sum
        self.total_priority=total_priority
        self.fuel_used=fuel_used 
        self.fuel_left=fuel_left
        self.fitness=self.evaluate()
        
    def evaluate(self):
       '''
       calculates fitness score of individual
       --------------------------------------
       
       Returns
       --------------------------------------
       float, fitness score
       '''
       if self.fuel_used==np.inf:
           fitness=0.8*self.total_priority
       else:
           fitness=0.8*self.total_priority+0.2*self.fuel_left
       return fitness
    
    def clone(self):
     '''
     clones a vessel individual
     --------------------------------
     
     Returns
     ---------------------------------
     Individual object, deep copy of this individual
     '''
     return Individual(
        chromosome=copy.deepcopy(self.chromosome),    
        generation=self.generation,
        violation_sum=self.violation_sum,
        total_priority=self.total_priority,
        fuel_left=self.fuel_left, 
        fuel_used=self.fuel_used
       )
    
    def __gt__(self, other):
        '''
        Greater than operater for individuals
        --------------------------------------------------------------------------
        Parameters
        ---------------------------------------------------------------------------
        :param other: individual object to compare this individual to
        
        Returns
        ---------------------------------------------------------------------------
        boolean, true if this individual is greater than other
        '''
        if  self.violation_sum==0 and other.violation_sum!=0:return True
        elif other.violation_sum==0 and self.violation_sum!=0:return False
        elif self.violation_sum!=0 and other.violation_sum!=0:
            if self.violation_sum<other.violation_sum : return True
            elif self.violation_sum>other.violation_sum:return False
            else:
                if self.fitness>other.fitness :return True
                else:return False
        else:
            if self.fitness>other.fitness :return True
            elif self.fitness<other.fitness :return False
            else:
                if self.fuel_used<other.fuel_used :return True
                else:return False

class Optimizer(object):
    '''Class for the genetic algorithm'''
    def __init__(self,prob,Rseed,Npop=100,PCross=0.75,PMut=0.15,Nelite=2):
        '''
        Summary
        -----------------------------------------------------------------------
        Optimizer Constructor
        
        Parameters
        ------------------------------------------------------------------------
        :param prob: VesselProblem object
        :param RSeed: int, Random seed for random number generator 
        :param Npop: int, Number of individuals in the population, needs to be an even integer 
        :param PCross: float,	Probability of crossover for a pair of two individuals 
        :param PMut: float, Probability of mutating an individual randomly
        :param Nelite: int,	Number of elite individuals to pass from one generation to the next
        '''
        #Initialize random number generator
        self.rng = Random(Rseed)
        
        self.prob = prob
        self.Npop=Npop
        self.PCross=PCross
        self.PMut=PMut
        self.Nwp=len(self.prob.waypoints)-2
        self.Nelite=Nelite
        
    def __initialize_population(self,size):
        '''
        Summary
        ------------------------------------------------
        Initialize the population for the vessel problem  
     
        
        Parameters
        -------------------------------------------------
        :param size: int, size of population
        
        Returns
        --------------------------------------------------
        List of Individuals: Individuals in initial population
        '''
        self.current_ident = 0
        initial_population = []
        wps_nums=[]
        for i in range(1,len(self.prob.waypoints)-1):
            wps_nums.append(i)
       
        for i in range(size):
            self.current_ident += 1 
            self.rng.shuffle(wps_nums)
            
            if self.Nwp>2:
             size=self.rng.randint(2,self.Nwp)
            else:
               size=self.Nwp
            chrom=[[0],[]]
            for i in range(0,size+2):
               speed=self.rng.choice(self.prob.speeds)
               chrom[1].append(speed)
            for i in range(0,size):
             chrom[0].append(wps_nums[i])
            chrom[0].append(len(self.prob.waypoints)-1)
            ind = self.make_vessel_individual(chrom)          
            initial_population.append(ind)
        return initial_population       
    
    def calculate_fuel(self,mission,edit_estimates=False):
        '''
        Summary
        ---------------------------------------------------------------------------
        returns the total fuel needed to reach all destinations in mission in order
       
        Parameters
        -----------------------------------------------------------------
        :param mission: mission object
        :param edit_estimates: bool, if true the arrival time estimates will be updated
     
        Returns
        -------------------------------------------------------------------
        float: total fuel used in kN
        '''
        if(len(mission.destinations)==0):
           return 0
        timestamp=self.prob.start_time
        total_time=0
        avg_speed=0
        total_distance=0
        for i in range(0,len(mission.destinations)-1):
           time_left=(mission.destinations[i].arrival_time-timestamp).total_seconds()/3600
           if edit_estimates and i!=0:
            mission.destinations[i].estimated_arrival_time=timestamp
           for time in mission.destinations[i].time_blacklist:
              if abs(time - timestamp) < timedelta(hours=6):
                 return np.inf
           if(time_left<0):
              mission.total_priority+=time_left
              
           g=geod.Inverse(mission.destinations[i].waypoint.lat, mission.destinations[i].waypoint.long,
                              mission.destinations[i+1].waypoint.lat,mission.destinations[i+1].waypoint.long)
           distance = g['s12'] / 1852
           self.prob.vessel.heading=g['azi1']
           old_speed=self.prob.vessel.speed
           self.prob.vessel.speed=mission.speeds[i]
           speed=self.prob.vessel.adjust_speed(mission.destinations[i].waypoint,distance,self.prob.start_time,timestamp)
           self.prob.vessel.speed=old_speed
           #avg_speed+=speed
           total_time += distance /speed
           total_distance+=distance
           timestamp+= timedelta(hours=distance /speed)
      
           if speed==-1:
            return np.inf; 
        if(edit_estimates):
            mission.destinations[len(mission.destinations)-1].estimated_arrival_time=timestamp
        if(total_time==0):
           return 0
        avg_speed=total_distance/total_time
        if(avg_speed==0):
           return 0
        total_fuel=self.prob.vessel.propSim1.runMachinery(avg_speed,self.prob.vessel,total_time)[0]
        
        return total_fuel
    
    def make_vessel_individual(self,chrom):
        '''
        Summary
        --------------------------------------
        makes a Individual from a chromosome
        
        Parameters
        --------------------------------------
         :param chrom: 2D list with 2 equal sized rows, chromosme representing destinations in a mission and vessel speeds 
        
        Returns
        ----------------------------------------------------------------------
        Individual: made from chrom
        '''
        from Simulation.vessel_sim_engine import  Mission
        destinations=[]
            
        for j in range(0, len(chrom[0])):
            destinations.append(self.prob.waypoints[chrom[0][j]])
        
        mission=Mission(destinations,chrom[1])
        
        fuel_used=self.calculate_fuel(mission)
        violation_sum=0
        norm_priority=(mission.total_priority)/self.prob.max_priority
        norm_fuel_left=(self.prob.vessel.fuel_weight-fuel_used)/self.prob.vessel.fuel_weight
        #if vessel doesn't have enough fuel to complete mision
        if fuel_used>self.prob.vessel.fuel_weight:
         violation_sum=(fuel_used-self.prob.vessel.fuel_weight)/self.prob.vessel.fuel_weight
            
        ind = Individual(chrom,1,violation_sum,norm_priority,norm_fuel_left,fuel_used) 
        return ind       
    
    def get_n_best(self, population):
        '''
        Summary
        -----------------------------------------------------
        Returns the NElite best individuals in the population
       
        Parameters
        -----------------------------------------------------
         :param population: list of individuals

        Returns
        -----------------------------------------------------
        list of individuals: contains NElite best individuals
        '''
        population.sort()
        n_best=[]
        for i in range(len(population)-self.Nelite,len(population)):
         n_best.append(population[i])
        return n_best
    
    def individual_comparison(self, ind1, ind2):
        '''
        Summary
        -----------------------------------------------------------------
        Compares two individuals using the prescribed constraint handling
        technique.
        
        Parameters
        -------------------------------------------------------------------
        :param ind1:   The first individual to be compared
        :param ind2:   The second individual to be compared
      
        Returns
        -------------------------------------------------------------------
        The better of the two individuals
        '''
        if(ind1>ind2):
            return ind1
        else: 
            return ind2
    
    def crossover(self, parent1, parent2):
        '''
        Summary
        ----------------------------------------
        Performs crossover on 2 chromosomes.
        
        Parameters
        -----------------------------------------
        :param parent 1: 2D list with 2 equal sized rows,  first chromosome
        :param parent 2: 2D list with 2 equal sized rows, second chromosome
        
        Returns
        ------------------------------------------
        list of two new individuals 
        '''
        shortest_len=0
        
        if(len(parent1[0])<len(parent2[0])): shortest_len=len(parent1[0])
        else:shortest_len=len(parent2[0])
        if(shortest_len<=2):
         ind1=self.make_vessel_individual(parent1)
         ind2=self.make_vessel_individual(parent2)
        
         return [ind1,ind2]
        n_wp=self.rng.randint(1,shortest_len-2)
        n_speed=self.rng.randint(1,shortest_len-2)
        chrom1=[[],[]]
        chrom2=[[],[]]
        
        #crossover waypoints
        for i in range(0,n_wp):
         if(parent1[0][i] not in chrom1[0]):
           chrom1[0].append(parent1[0][i])
           
         if(parent2[0][i] not in chrom2[0]):
            chrom2[0].append(parent2[0][i])
            
        for i in range(n_wp,len(parent2[0])):
            if(parent2[0][i] not in chrom1[0]):
                chrom1[0].append(parent2[0][i])
                
        for i in range(n_wp,len(parent1[0])):
             if(parent1[0][i] not in chrom2[0]):
              chrom2[0].append(parent1[0][i])
              
         #crossover speeds
        for i in range(0,n_speed):
           chrom1[1].append(parent1[1][i])
           chrom2[1].append(parent2[1][i])
        for i in range(n_speed,len(parent2[1])):
                chrom1[1].append(parent2[1][i])
        for i in range(n_speed,len(parent1[1])):
              chrom2[1].append(parent1[1][i])
          
          #pad speeds to match waypoints
        def pad_speeds(speeds, target_len, source):
           if len(speeds) > target_len:
             return speeds[:target_len]
           while len(speeds) < target_len:
            speeds.append(self.rng.choice(source))
           return speeds

        chrom1[1] = pad_speeds(chrom1[1], len(chrom1[0]), parent1[1] + parent2[1])
        chrom2[1] = pad_speeds(chrom2[1], len(chrom2[0]), parent1[1] + parent2[1])
        if(chrom1[0][0] != 0 or chrom1[0][len(chrom1[0])-1]!=len(self.prob.waypoints)-1):
           logger.error("chrom1 start/end error during crossover")
        if(chrom2[0][0] != 0 or chrom2[0][len(chrom2[0])-1]!=len(self.prob.waypoints)-1):
           logger.error("chrom 2 start/end error during crossover")
        if(len(chrom1[0])!=len(chrom1[1])or len(chrom2[0])!=len(chrom2[1])):
           logger.error("chromosome length error durig crossover")
        ind1=self.make_vessel_individual(chrom1)
        ind2=self.make_vessel_individual(chrom2)
        
        return [ind1,ind2]
    
    def partial_reverse_waypoints(self, chrom):
        '''
        Summary
        ---------------------------------------------------------------------
        Mutation that reverses a random section of a chromosome but doesn't change the speeds.
       
        Parameters
        ---------------------------------------------------------------------
        :param chrom: 2D list with 2 equal sized rows, chromosome to be mutated
        
        Returns
        ---------------------------------------------------------------------
        Individual: new individual with mutated chromosome
        '''
        if(len(chrom[0])<=2):
           return self.make_vessel_individual(chrom)
        start1=self.rng.randint(1,len(chrom[0])-2)
        stop1=self.rng.randint(start1+1,len(chrom[0])-1)
        chrom[0][start1:stop1] = chrom[0][start1:stop1][::-1]
        chrom[1][start1:stop1] = chrom[1][start1:stop1][::-1]
        if(chrom[0][0] != 0 or chrom[0][len(chrom[0])-1]!=len(self.prob.waypoints)-1):
           logger.error("chromosome start/end error on partial reverse")
        return self.make_vessel_individual(chrom)
    
    def drop_waypoint(self, chrom):
       '''
        Summary
        ---------------------------------------------------------------------
        Mutation that removes a random waypoint index from an individual's chromosome.
        
        Parameters
        ---------------------------------------------------------------------
        :param chrom: 2D list with 2 equal sized rows, chromosome to be mutated
       
        Returns
        ---------------------------------------------------------------------
        Individual: new individual with mutated chromosome
        '''
       if(len(chrom[0])<=2):
          return self.make_vessel_individual(chrom)
       drop=self.rng.randint(1,len(chrom[0])-2) 
       del chrom[0][drop] 
       del chrom[1][drop]
       if(chrom[0][0] != 0 or chrom[0][len(chrom[0])-1]!=len(self.prob.waypoints)-1):
           logger.error("chromosome start/end error during drop waypoint")
       return self.make_vessel_individual(chrom)               
    
    def add_waypoint(self, chrom):
       '''
        Summary
        ----------------------------------------------------------------------
        mutation that adds a random waypoint index to the end of a chromosome
       
        Parameters
        ---------------------------------------------------------------------
        :param chrom: 2D list of ints with 2 equal sized rows, chromosome to be mutated
        
        Returns
        ---------------------------------------------------------------------
        Individual: new individual with mutated chromosome
       '''
      
       add_ind=self.rng.randint(1,len(self.prob.waypoints)-2) 
       if(add_ind not in chrom[0]):
        chrom[0].insert(len(chrom[0])-1,add_ind)
        chrom[1].insert(len(chrom[1])-2,self.rng.choice(self.prob.speeds))
       if(chrom[0][0] != 0 or chrom[0][len(chrom[0])-1]!=len(self.prob.waypoints)-1):
           logger.error("chromosome start/end error on add waypoint")
       return self.make_vessel_individual(chrom)
    
    def replace_waypoint(self,chrom):
       '''
        Summary
        --------------------------------------------------------------------
        mutation that replaces a random index in the chromosome with a random waypoint index
        
        Parameters
        ---------------------------------------------------------------------
        :param chrom: 2D list with 2 equal sized rows, chromosome to be mutated
        
        Returns
        ---------------------------------------------------------------------
        Individual: new individual with mutated chromosome
       '''
       if(len(chrom[0])<=2):
          return self.make_vessel_individual(chrom)
       replace_ind=self.rng.randint(1, len(chrom[0])-2)
       add_ind=self.rng.randint(1,len(self.prob.waypoints)-2) 
       if(add_ind not in chrom[0]):
        chrom[0][replace_ind]=add_ind
        chrom[1][replace_ind]=self.rng.choice(self.prob.speeds)
        if(chrom[0][0] != 0 or chrom[0][len(chrom[0])-1]!=len(self.prob.waypoints)-1):
           logger.error("chromosome start/end error on replace waypoint")
       return self.make_vessel_individual(chrom)
    
    def replace_speed(self,chrom):
       '''
        Summary
        --------------------------------------------------------------------
        mutation that replaces a random speed the chromosome with a random speed
        
        Parameters
        ---------------------------------------------------------------------
        :param chrom: 2D list with 2 equal sized rows, chromosome to be mutated
        
        Returns
        ---------------------------------------------------------------------
        Individual: new individual with mutated chromosome
       '''
       if(len(chrom[0])<=2):
          return self.make_vessel_individual(chrom)
          
       replace_ind=self.rng.randint(1, len(chrom[0])-2)
       chrom[1][replace_ind]=self.rng.choice(self.prob.speeds)
       if(chrom[0][0] != 0 or chrom[0][len(chrom[0])-1]!=len(self.prob.waypoints)-1):
           logger.error("chromosome start/end error on replace speed")
       return self.make_vessel_individual(chrom)
   
    def two_pass_tournament_selection(self, pop):
      ''' 
      Summary
      --------------------------------------------------------------------------------
      The list of individuals is randomly shuffled.  Adjecent individuals are compared 
      and the better individual from each comparison is added to the selection list 
      This process is repeated a second time to fill the list.
      
      Parameters
      --------------------------------------------------------------------------------
      :param pop: list of individuals
      
      Returns
      --------------------------------------------------------------------------------
      list of individuals: the selection population
      '''
      selection=[]
    
      for i in range(0,2):
       self.rng.shuffle(pop)
       for j in range(0,(len(pop))//2):
          if pop[j]>pop[j+1]:
             selection.append(pop[j])
             
          else:
             selection.append(pop[j+1])
             
      return selection
    
    def calc_mean_score(self,pop):
       '''
       Summary
       -----------------------------------------
       calculates the mean priority score in pop
     
       Parameters
       -----------------------------------------
       pop: list of individuals
       
       Returns
       ----------------------------------------
       float: mean 
       '''
       total=0.0
       for ind in pop:
          total+=ind.fitness
       return total/len(pop)
    
    def local_search_speed(self, individual, positions_to_try=5):
      """
       Summary
       ----------------------------------------------
       Perform local search on a discrete speed list.
       Tries up to positions_to_try random segments.
       
       Parameters
       -----------------------------------------------
       :param individual: individual object, individual to search
       :param positions_to_try: int, number of speed positions to search in chrom
      """
      best_ind = individual.clone()
      best_fitness = best_ind.fitness
      allowed_speeds = self.prob.speeds  

    # Randomly pick which speed indices to try
      num_speeds = len(best_ind.chromosome[1])
      positions = self.rng.sample(range(num_speeds), k=min(positions_to_try, num_speeds))

      for i in positions:
        current_speed = best_ind.chromosome[1][i]
        for alt_speed in allowed_speeds:
            if alt_speed == current_speed:
                continue
            new_chromosome = [row[:] for row in best_ind.chromosome]  
            new_chromosome[1][i] = alt_speed
            new_ind = self.make_vessel_individual(new_chromosome)

            if new_ind.fitness > best_fitness and new_ind.violation_sum == 0:
                best_ind = new_ind
                best_fitness = new_ind.fitness
                break  # Accept first improvement

      return best_ind
    
    def calc_mean_score_no_violations(self,pop):
       '''
       Summary
       ------------------------------------------------
       Calculates the mean priority score in pop. 
       Excludes individuals with a constraint violation.
      
       Parameters
       ------------------------------------------------
       :param pop: list of individuals
       
       Returns
       ------------------------------------------------
       float: mean 
       '''
       total=0.0
       count=0
       for ind in pop:
          if ind.violation_sum==0:
           total+=ind.fitness
           count+=1
       if count==0:
          return 0
       return total/count
   
    def calc_mean_mission_length(self,pop):
       '''
       Summary
       -----------------------------------------
       calculates the mean mission length in pop
       
       Parameters
       -----------------------------------------
       :param pop: list of individuals
       
       Returns
       ----------------------------------------
       float: mean 
       '''
       total=0.0
       for ind in pop:
        total+=len(ind.chromosome)
       return total/len(pop)
    
    def find_percent_ind(self,pop,ind):
       '''
       Summary
       --------------------------------------------
       Finds the percentage of pop that contains ind.
       
       Parameters
       -------------------------------------------
       :param pop: list of Individual Objects, population
       :param ind: Individual object
       
       Returns:
       --------------------------------------------
       float, percentage of pop that contains ind
       '''
       count=0
       for i in pop:
          if(i.chromosome[0]==ind.chromosome[0] and i.chromosome[1]==ind.chromosome[1]):
             count+=1
       return count/len(pop)*100
    
    def run(self, max_generations=50):
        '''
        Summary
        ---------------------------------------------------------------
        Runs the GA and evolves a population "max_generations" times
        
        Parameters
        ---------------------------------------------------------------
        :param max_generations: int, number of times the population evolves
        
        Returns
        -------
        opt_dests: list of destination objects, destinations that make up the 
        optimal mission, also includes 
        the vessel's start and end ponts
        speeds: list floats, vessel speeds between each destination
        fuel_left: floatsestimated amount of fuel remaining after completing the mission 
        fitness: float, fitness score of the optimal mission
        '''
        from Simulation.vessel_sim_engine import Mission

        scores = []
        generations = []
        mean_scores = []
        mean_scores_no_violation = []
        mean_mission_length = []
        percents_bests=[]
        
        self.current_generation = 0
        if(self.Nwp<1):
           return [self.prob.start,self.prob.end],[self.prob.speeds[0],self.prob.speeds[0]],self.prob.vessel.fuel_weight,0
        pop = self.__initialize_population(self.Npop)

    
        elite = [e.clone() for e in self.get_n_best(pop)]

        while self.current_generation < max_generations:
          #print(self.current_generation)
          generations.append(self.current_generation)
          self.current_generation += 1
          if self.current_generation % 5 == 0:
            pop[0] = self.__initialize_population(1)[0]
            percents_bests.append((self.find_percent_ind(pop,elite[self.Nelite - 1]),self.current_generation))
          selection = self.two_pass_tournament_selection(pop)
          next_gen = []

          while len(selection) > 1:
            parent1 = selection.pop()
            parent2 = selection.pop()
            if parent1.chromosome == parent2.chromosome:
                if(parent2>parent1):
                  parent1 = self.__initialize_population(1)[0]
                else:
                   parent2 = self.__initialize_population(1)[0]
            children = [parent1.clone(), parent2.clone()]

            if self.rng.random() < self.PCross:
                
                children = self.crossover(children[0].chromosome, children[1].chromosome)
                mutations = [
                self.partial_reverse_waypoints,
                self.drop_waypoint,
                self.add_waypoint,
                self.replace_waypoint,
                self.replace_speed,
                 ]
                # Mutation
                for i in range(2):
                  if self.rng.random() < self.PMut:
                    mutation_func = self.rng.choice(mutations)
                    children[i] = mutation_func(children[i].chromosome)
         

            next_gen += children

          next_gen.sort()
          num_replaced = 0

       
          for e in elite:
             if e not in next_gen:
                next_gen[num_replaced] = e.clone()
                num_replaced += 1

          pop = next_gen 
          elite = [e.clone() for e in self.get_n_best(next_gen)]
          if self.current_generation % 5 == 0:
           for e in elite: 
             e=self.local_search_speed(e,10)
          elite.sort()

          scores.append(elite[self.Nelite - 1].fitness)
          mean_scores.append(self.calc_mean_score(next_gen))
          mean_scores_no_violation.append(self.calc_mean_score_no_violations(next_gen))
          mean_mission_length.append(self.calc_mean_mission_length(next_gen))

           
        opt_dests=[]
        if elite[self.Nelite-1].violation_sum==0:
         for e in elite[self.Nelite-1].chromosome[0]:
          opt_dests.append(self.prob.waypoints[e])
        else:
           opt_dests=[self.prob.waypoints[elite[self.Nelite-1].chromosome[0][0]],self.prob.waypoints[elite[self.Nelite-1].chromosome[0][-1]]]
           elite[self.Nelite-1].fitness=0
       # opt_dests.append(self.prob.end)
        fuel=self.calculate_fuel(Mission(opt_dests,elite[self.Nelite-1].chromosome[1]),True)
    
        
        fuel_left=self.prob.vessel.fuel_weight-fuel
        
        print(f"predicted fuel used {fuel}")
        print(f"predicted fuel remaining {fuel_left}")
        print(f"fitness {elite[self.Nelite-1].fitness}")
        '''
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(generations,scores,label='Best Fitness Score')
        ax.plot(generations,mean_scores,label='Mean Fitness Score')
        ax.plot(generations,mean_scores_no_violation,label='Mean Fitness No Violation')
        #plt.plot(generations,mean_mission_len,label='Mean Mission Length')
        ax.set_title("Mission Priority Scores Vs. Generation")
        ax.set_xlabel('generation')
        ax.set_ylabel('proportion of max score')
        ax.legend()
        plt.show()
     
        records=[]
        for entry in percents_bests:
           records.append({
            "Vessel": self.prob.vessel.name,
            "Seed": 0,
            "Generation": entry[1],
            "Percent Best Ind": entry[0]
        })
        '''
        return opt_dests,elite[self.Nelite-1].chromosome[1],fuel_left,elite[self.Nelite-1].fitness
