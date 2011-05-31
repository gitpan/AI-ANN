#!/usr/bin/perl
package AI::ANN::Evolver;
BEGIN {
  $AI::ANN::Evolver::VERSION = '0.002';
}
# ABSTRACT: an evolver for an artificial neural network simulator

use strict;
use warnings;
use AI::ANN;
use Storable qw(dclone);


sub new {
	my $proto = shift;
	my $class = ref($proto) || $proto;
	my $self = {};
	my $data = shift || {};
	$self->{'max_value'}=$data->{'max_value'} || 1;
	$self->{'min_value'}=$data->{'min_value'} || 0;
	$self->{'mutation_chance'}=$data->{'mutation_chance'} || 0;
	$self->{'mutation_amount'}=$data->{'mutation_amount'} || 0;
	$self->{'add_link_chance'}=$data->{'add_link_chance'} || 0;
	$self->{'kill_link_chance'}=$data->{'kill_link_chance'} || 0;
	$self->{'sub_crossover_chance'}=$data->{'sub_crossover_chance'} || 0;
	bless($self, $class);
	return $self;
}


sub crossover {
	my $self = shift;
	my $network1 = shift;
	my $network2 = shift;
	my $class = ref($network1);
	my $inputcount = $network1->get_input_count();
	my $minvalue = $network1->get_minvalue();
	my $maxvalue = $network1->get_maxvalue();
	my $afunc = $network1->get_afunc();
	# They better have the same number of inputs
	$inputcount == $network2->get_input_count() || return -1; 
	my $networkdata1 = $network1->get_internals();
	my $networkdata2 = $network2->get_internals();
	my $neuroncount = $#{$networkdata1};
	# They better also have the same number of neurons
	$neuroncount == $#{$networkdata2} || return -1;
	my $networkdata3 = [];

	for (my $i = 0; $i <= $neuroncount; $i++) {
		if (rand() < $self->{'sub_crossover_chance'}) {
			$networkdata3->[$i] = { 'inputs' => {}, 'neurons' => {} };
			$networkdata3->[$i]->{'iamanoutput'} = 
				$networkdata1->[$i]->{'iamanoutput'};
			for (my $j = 0; $j < $inputcount; $j++) {
				$networkdata3->[$i]->{'inputs'}->{$j} = 
					(rand() > 0.5) ?
					$networkdata1->[$i]->{'inputs'}->{$j} :
					$networkdata2->[$i]->{'inputs'}->{$j};
				# Note to self: Don't get any silly ideas about dclone()ing 
				# these, that's a good way to waste half an hour debugging.
			}
			for (my $j = 0; $j <= $neuroncount; $j++) {
				$networkdata3->[$i]->{'neurons'}->{$j} =
					(rand() > 0.5) ?
					$networkdata1->[$i]->{'neurons'}->{$j} :
					$networkdata2->[$i]->{'neurons'}->{$j};
			}
		} else {
			$networkdata3->[$i] = dclone(
				(rand() > 0.5) ?
				$networkdata1->[$i] :
				$networkdata2->[$i] );
		}		
	}
	my $network3 = $class->new ({ 'inputs' => $inputcount, 
								  'data' => $networkdata3,
								  'minvalue' => $minvalue,
								  'maxvalue' => $maxvalue,
								  'afunc' => $afunc});
	return $network3;
}


sub mutate {
	my $self = shift;
	my $network = shift;
	my $class = ref($network);
	my $networkdata = $network->get_internals();
	my $inputcount = $network->get_input_count();
	my $minvalue = $network->get_minvalue();
	my $maxvalue = $network->get_maxvalue();
	my $afunc = $network->get_afunc();
	my $neuroncount = $#{$networkdata}; # BTW did you notice that this 
										# isn't what it says it is?
	$networkdata = dclone($networkdata); # For safety.
	for (my $i = 0; $i <= $neuroncount; $i++) {
		# First each input/neuron pair
		for (my $j = 0; $j < $inputcount; $j++) {
			# I think the following like is right, but it's also midnight...
			my $weight = $networkdata->[$i]->{'inputs'}->{$j};
			if (defined $weight && $weight != 0) {
				if (rand() < $self->{'mutation_chance'}) {
					$weight += (rand() * 2 - 1) * $self->{'mutation_amount'};
					if ($weight > $self->{'max_value'}) { 
						$weight = $self->{'max_value'};
					}
					if ($weight < $self->{'min_value'}) { 
						$weight = $self->{'min_value'} + 0.000001;
					}
				} 
				if (abs($weight) < $self->{'mutation_amount'}) {
					if (rand() < $self->{'kill_link_chance'}) {
						$weight = undef;
					}
				}
			} else {
				if (rand() < $self->{'add_link_chance'}) {
					$weight = rand() * $self->{'mutation_amount'};
					# We want to Do The Right Thing. Here, that means to 
					# detect whether the user is using weights in (0, x), and
					# if so make sure we don't accidentally give them a 
					# negative weight, because that will become 0.000001. 
					# Instead, we'll generate a positive only value at first 
					# (it's easier) and then, if the user will accept negative 
					# weights, we'll let that happen.
					if ($self->{'min_value'} < 0) {
						($weight *= 2) -= $self->{'mutation_amount'};
					}
					# Of course, we have to check to be sure...
					if ($weight > $self->{'max_value'}) { 
						$weight = $self->{'max_value'};
					}
					if ($weight < $self->{'min_value'}) { 
						$weight = $self->{'min_value'} + 0.000001;
					}
					# But we /don't/ need to to a kill_link_chance just yet.
				}
			}
			# This would be a bloody nightmare if we hadn't done that dclone 
			# magic before. 
			$networkdata->[$i]->{'inputs'}->{$j} = $weight;
		}
		# Now each neuron/neuron pair
		for (my $j = 0; $j <= $neuroncount; $j++) {
		# As a reminder to those cursed with the duty of maintaining this code:
		# This should be an exact copy of the code above, except that 'inputs' 
		# would be replaced with 'neurons'. 
			my $weight = $networkdata->[$i]->{'neurons'}->{$j};
			if (defined $weight && $weight != 0) {
				if (rand() < $self->{'mutation_chance'}) {
					$weight += (rand() * 2 - 1) * $self->{'mutation_amount'};
					if ($weight > $self->{'max_value'}) { 
						$weight = $self->{'max_value'};
					}
					if ($weight < $self->{'min_value'}) { 
						$weight = $self->{'min_value'} + 0.000001;
					}
				} 
				if (abs($weight) < $self->{'mutation_amount'}) {
					if (rand() < $self->{'kill_link_chance'}) {
						$weight = undef;
					}
				}
			} else {
				if (rand() < $self->{'add_link_chance'}) {
					$weight = rand() * $self->{'mutation_amount'};
					# We want to Do The Right Thing. Here, that means to 
					# detect whether the user is using weights in (0, x), and
					# if so make sure we don't accidentally give them a 
					# negative weight, because that will become 0.000001. 
					# Instead, we'll generate a positive only value at first 
					# (it's easier) and then, if the user will accept negative 
					# weights, we'll let that happen.
					if ($self->{'min_value'} < 0) {
						($weight *= 2) -= $self->{'mutation_amount'};
					}
					# Of course, we have to check to be sure...
					if ($weight > $self->{'max_value'}) { 
						$weight = $self->{'max_value'};
					}
					if ($weight < $self->{'min_value'}) { 
						$weight = $self->{'min_value'} + 0.000001;
					}
					# But we /don't/ need to to a kill_link_chance just yet.
				}
			}
			# This would be a bloody nightmare if we hadn't done that dclone 
			# magic before. 
			$networkdata->[$i]->{'neurons'}->{$j} = $weight;
		}
		# That was rather tiring, and that's only for the first neuron!!
	}
	# All done. Let's pack it back into an object and let someone else deal
	# with it.
	$network = $class->new ({ 'inputs' => $inputcount, 
								 'data' => $networkdata,
								 'minvalue' => $minvalue,
								 'maxvalue' => $maxvalue,
								 'afunc' => $afunc});
	return $network;
}

1;
		

__END__
=pod

=head1 NAME

AI::ANN::Evolver - an evolver for an artificial neural network simulator

=head1 VERSION

version 0.002

=head1 METHODS

=head2 new

AI::ANN::Evolver->new( { mutation_chance => $mutationchance, 
	mutation_amount => $mutationamount, add_link_chance => $addlinkchance, 
	kill_link_chance => $killlinkchance, sub_crossover_chance => 
	$subcrossoverchance, min_value => $minvalue, max_value => $maxvalue } )

All values have a sane default.

mutation_chance is the chance that calling mutate() will add a random value
	on a per-link basis. It only affects existing (nonzero) links. 
mutation_amount is the maximum change that any single mutation can introduce. 
	It affects the result of successful mutation_chance rolls, the maximum 
	value after an add_link_chance roll, and the maximum strength of a link 
	that can be deleted by kill_link_chance rolls. It can either add or 
	subtract.
add_link_chance is the chance that, during a mutate() call, each pair of 
	unconnected neurons or each unconnected neuron => input pair will 
	spontaneously develop a connection. This should be extremely small, as
	it is not an overall chance, put a chance for each connection that does
	not yet exist. If you wish to ensure that your neural net does not become 
	recursive, this must be zero. 
kill_link_chance is the chance that, during a mutate() call, each pair of 
	connected neurons with a weight less than mutation_amount or each 
	neuron => input pair with a weight less than mutation_amount will be
	disconnected. If add_link_chance is zero, this should also be zero, or 
	your network will just fizzle out.
sub_crossover_chance is the chance that, during a crossover() call, each 
	neuron will, rather than being inherited fully from each parent, have 
	each element within it be inherited individually.
min_value is the smallest acceptable weight. It must be less than or equal to 
	zero. If a value would be decremented below min_value, it will instead 
	become an epsilon above min_value. This is so that we don't accidentally 
	set a weight to zero, thereby killing the link.
max_value is the largest acceptable weight. It must be greater than zero.

=head2 crossover

$evolver->crossover( $network1, $network2 )

Returns a $network3 consisting of the shuffling of $network1 and $network2
As long as the same neurons in network1 and network2 are outputs, network3 
	will always have those same outputs.
This method, at least if the sub_crossover_chance is nonzero, expects neurons 
	to be labeled from zero to n. 

=head2 mutate

$evolver->mutate($network)

Returns a version of $network mutated according to the parameters set for 
	$evolver, followed by a series of counters. The original is not modified. 
	The counters are, in order, the number of times we compared against the 
	following thresholds: mutation_chance, kill_link_chance, add_link_chance. 
	This is useful if you want to try to normalize your probabilities. For 
	example, if you want to make links be killed about as often as they are 
	added, keep a running total of the counters, and let:
	$kill_link_chance = $add_link_chance * $add_link_counter / $kill_link_counter
	This will probably make kill_link_chance much larger than add_link_chance, 
	but in doing so will make links be added at overall the same rate as they 
	are killed. Since new links tend to be killed particularly quickly, it may 
	be wise to add an additional optional multiplier to mutation_amount just 
	for new links.

=head1 AUTHOR

Dan Collins <DCOLLINS@cpan.org>

=head1 COPYRIGHT AND LICENSE

This software is Copyright (c) 2011 by Dan Collins.

This is free software, licensed under:

  The GNU General Public License, Version 3, June 2007

=cut

