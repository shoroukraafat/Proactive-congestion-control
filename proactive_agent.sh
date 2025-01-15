#! /bin/bash
for i in {1..1000}
do
	./waf --run "scratch/proactive_agent/proactive_agent --RunNum=$(($i))"
done
