#!/bin/bash

COLOR_OFF='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
BROWN='\033[0;33m'
BLUE='\033[0;34m'



if [ $1 == '--start_dev' ]
then

  printf "${GREEN}Starting development enviroment${COLOR_OFF}\n"
  docker-compose -f Docker/docker-compose.yaml up -d --build
  sudo chmod -R go+w Data
  sudo chmod -R go+w Models
  sudo chmod -R go+w Notebooks
  sudo chmod -R go+w Scripts

elif [ $1 == '--close_dev' ]
then

  printf "${RED}Closing development enviroment${COLOR_OFF}\n"
  docker-compose -f Docker/docker-compose.yaml down

elif [ $1 == '--start_prod' ]
then

  printf "${GREEN}Starting service${COLOR_OFF}\n"
  docker build -t cristiancristanchot/hedwig_ai:latest Docker/Hedwig
  docker run --name hedwig_container -d --gpus all -p 7860:7860 cristiancristanchot/hedwig_ai:latest

elif [ $1 == '--close_prod' ]
then

  printf "${RED}Closing service${COLOR_OFF}\n"
  docker stop hedwig_container
  docker rm hedwig_container

elif [ $1 == '--help' ]
then

  printf "${GREEN}Options:
  	--start_dev	Start development enviroment
  	--close_dev	Close development enviroment
  	${COLOR_OFF}\n"

else
  printf "${RED}Error: Invalid option. Type '$0 --help' for available options.${COLOR_OFF}\n"
  exit 1
fi
