#!/usr/bin/perl
package AI::ANN;
BEGIN {
  $AI::ANN::VERSION = '0.006';
}
use strict;
use warnings;

# ABSTRACT: an artificial neural network simulator

use Moose;

use AI::ANN::Neuron;
use Storable qw(dclone);


has 'input_count' => (is => 'ro', isa => 'Int', required => 1);
has 'outputneurons' => (is => 'ro', isa => 'ArrayRef[Int]', required => 1);
has 'network' => (is => 'ro', isa => 'ArrayRef[HashRef]', required => 1);
# network is an arrayref of hashrefs. Each hashref is:
# object => AI::ANN::Neuron
# and has several other elements
has 'inputs' => (is => 'ro', isa => 'ArrayRef[Int]');
has 'minvalue' => (is => 'rw', isa => 'Int', default => 0);
has 'maxvalue' => (is => 'rw', isa => 'Int', default => 1);
has 'afunc' => (is => 'rw', isa => 'CodeRef', default => sub {sub {shift}});

around BUILDARGS => sub {
	my $orig = shift;
	my $class = shift;
	my %data;
	if ( @_ == 1 && ref $_[0] eq 'HASH' ) {
		%data = %{$_[0]};
	} else {
		%data = @_;
	}
	if (exists $data{'inputs'} && not exists $data{'input_count'}) {
		$data{'input_count'} = $data{'inputs'};
		delete $data{'inputs'}; # inputs is used later for the actual 
								# values of the inputs.
	} 
	my $neuronlist = $data{'data'};
	$data{'outputneurons'} = [];
	$data{'network'} = [];
	for (my $i = 0; $i <= $#{$neuronlist} ; $i++) {
		push @{$data{'outputneurons'}}, $i
			if $neuronlist->[$i]->{'iamanoutput'};
		my @pass = (
				$i, 
				$neuronlist->[$i]->{'inputs'}, 
				$neuronlist->[$i]->{'neurons'} );
		push @pass, $neuronlist->[$i]->{'eta_inputs'}, 
			$neuronlist->[$i]->{'eta_neurons'}
			if defined $neuronlist->[$i]->{'eta_neurons'};
		$data{'network'}->[$i]->{'object'} = 
			new AI::ANN::Neuron( @pass );
	}
	delete $data{'data'};
	return $class->$orig(%data);
};


sub execute {
	my $self = shift;
	my $inputs = $self->{'inputs'} = shift;
	my $neurons = {};
	my $net = $self->{'network'}; # For less typing
	for (my $i = 0; $i <= $#{$net}; $i++) {
		delete $net->[$i]->{'done'};
		delete $net->[$i]->{'state'};
	}
	my $progress = 0;
	do {
		$progress = 0;
		for (my $i = 0; $i <= $#{$net}; $i++) {
			if ($net->[$i]->{'done'}) {next}
			if ($net->[$i]->{'object'}->ready($inputs, $neurons)) {
				my $potential = $net->[$i]->{'object'}->execute($inputs, $neurons);
				$potential = &{$self->{'afunc'}}($potential);
				$potential = $self->{'maxvalue'} if $potential > $self->{'maxvalue'};
				$potential = $self->{'minvalue'} if $potential < $self->{'minvalue'};
				$neurons->{$i} = $net->[$i]->{'state'} = $potential;
				$net->[$i]->{'done'} = 1;
				$progress++;
			}
		}
	} while ($progress); # If the network is feed-forward, we are now finished.
	
	my @notdone = grep {not (defined $net->[$_]->{'done'} &&
					$net->[$_]->{'done'} == 1)} 0..$#{$net};
	my %notdone; # Apparently Perl gets confused if I my the next line
	@notdone{@notdone} = undef; # %notdone is now a hash with a key for each 
								# 'notdone' neuron.
	if ($#notdone > 0) { #This is the part where we deal with loops and bad things
		my $maxerror = 0;
		my $loopcounter = 1;
		while (1) {
			foreach my $i (keys %notdone) { # Only bother iterating over the
											# ones we couldn't solve exactly
				# We don't care if it's ready now, we're just going to interate
				# until it stabilizes.
				if (not defined $neurons->{$i} && $i <= $#{$net}) {
					# Fixes warnings about uninitialized values, but we make 
					# sure $i is valid first.
					$neurons->{$i} = 0;
				}
				my $potential = $net->[$i]->{'object'}->execute($inputs, $neurons);
				$potential = &{$self->{'afunc'}}($potential);
				$potential = $self->{'maxvalue'} if $potential > $self->{'maxvalue'};
				$potential = $self->{'minvalue'} if $potential < $self->{'minvalue'};
				$notdone{$i} = $net->[$i]->{'state'} = $potential;
				# We want to know the absolute change
				if (abs($neurons->{$i}-$notdone{$i})>$maxerror) {
					$maxerror = abs($neurons->{$i}-$notdone{$i});
				}
			}
			foreach my $i (keys %notdone) { 
				# Update $neurons, since that is what gets passed to execute
				$neurons->{$i}=$notdone{$i};
			}
			if ($maxerror < 0.0001 && $loopcounter >= 5) {last}
			$loopcounter++;
			$maxerror=0;
		}
	}
	
	# Ok, hopefully all the neurons have happy values by now.
	# Get the output values for neurons corresponding to outputneurons
	my @output = map {$net->[$_]->{'state'}} @{$self->{'outputneurons'}};
	return \@output;
}


sub get_state {
	my $self = shift;
	my $net = $self->{'network'}; # For less typing
	my @neurons = map {$net->[$_]->{'state'}} 0..$#{$self->{'network'}};
	my @output = map {$net->[$_]->{'state'}} @{$self->{'outputneurons'}};
	
	return $self->{'inputs'}, \@neurons, \@output;
}


sub get_internals {
	my $self = shift;
	my $retval = [];
	for (my $i = 0; $i <= $#{$self->{'network'}}; $i++) {
		$retval->[$i] = { iamanoutput => 0,
						  inputs => $self->{'network'}->[$i]->{'object'}->inputs(),
						  neurons => $self->{'network'}->[$i]->{'object'}->neurons(),
						  eta_inputs => $self->{'network'}->[$i]->{'object'}->eta_inputs(),
						  eta_neurons => $self->{'network'}->[$i]->{'object'}->eta_neurons()
						  };
	}
	foreach my $i (@{$self->{'outputneurons'}}) {
		$retval->[$i]->{'iamanoutput'} = 1;
	}
	return dclone($retval); # Dclone for safety.
}


sub readable {
	my $self = shift;
	my $retval = "This network has ". $self->{'inputcount'} ." inputs and ".
					scalar(@{$self->{'network'}}) ." neurons.\n";
	for (my $i = 0; $i <= $#{$self->{'network'}}; $i++) {
		$retval .= "Neuron $i\n";
		while (my ($k, $v) = each %{$self->{'network'}->[$i]->{'object'}->inputs()}) {
			$retval .= "\tInput from input $k, weight is $v\n";
		}
		while (my ($k, $v) = each %{$self->{'network'}->[$i]->{'object'}->neurons()}) {
			$retval .= "\tInput from neuron $k, weight is $v\n";
		}
		if (map {$_ == $i} $self->{'outputneurons'}) {
			$retval .= "\tThis neuron is a network output\n";
		}
	}
	return $retval;
}

1;

__END__
=pod

=head1 NAME

AI::ANN - an artificial neural network simulator

=head1 VERSION

version 0.006

=head1 SYNOPSIS

AI::ANN is an artificial neural network simulator. It differs from existing 
solutions in that it fully exposes the internal variables and allows - and 
forces - the user to fully customize the topology and specifics of the 
produced neural network. If you want a simple solution, you do not want this 
module. This module was specifically written to be used for a simulation of 
evolution in neural networks, not training. The traditional 'backprop' and 
similar training methods are not (currently) implemented. Rather, we make it 
easy for a user to specify the precise layout of their network (including both 
topology and weights, as well as many parameters), and to then retrieve those 
details. The purpose of this is to allow an additional module to then tweak 
these values by a means that models evolution by natural selection. The 
canonical way to do this is the included AI::ANN::Evolver, which allows 
the addition of random mutations to individual networks, and the crossing of 
two networks. You will also, depending on your application, need a fitness 
function of some sort, in order to determine which networks to allow to 
propagate. Here is an example of that system.

use AI::ANN;
my $network = new AI::ANN ( input_count => $inputcount, data => \@neuron_definition );
my $outputs = $network->execute( \@inputs ); # Basic network use
use AI::ANN::Evolver;
my $handofgod = new AI::ANN::Evolver (); # See that module for calling details
my $network2 = $handofgod->mutate($network); # Random mutations
# Test an entire 'generation' of networks, and let $network and $network2 be
# among those with the highest fitness function in the generation.
my $network3 = $handofgod->crossover($network, $network2);
# Perhaps mutate() each network either before or after the crossover to 
# introduce variety.

We elected to do this with a new module rather than by extending an existing 
module because of the extensive differences in the internal structure and the 
interface that were necessary to accomplish these goals. 

=head1 METHODS

=head2 new

ANN::new(input_count => $inputcount, data => [{ iamanoutput => 0, inputs => {$inputid => $weight, ...}, neurons => {$neuronid => $weight}}, ...])

input_count is number of inputs.
data is an arrayref of neuron definitions.
The first neuron with iamanoutput=1 is output 0. The second is output 1.
I hope you're seeing the pattern...
minvalue is the minimum value a neuron can pass. Default 0.
maxvalue is the maximum value a neuron can pass. Default 1.
afunc is a reference to the activation function. It should be simple and fast.
    The activation function is processed /before/ minvalue and maxvalue.

=head2 execute

$network->execute( [$input0, $input1, ...] )

Runs the network for as many iterations as necessary to achieve a stable
network, then returns the output. 
We store the current state of the network in two places - once in the object,
for persistence, and once in $neurons, for simplicity. This might be wrong, 
but I couldn't think of a better way.

=head2 get_state

$network->get_state()

Returns three arrayrefs, [$input0, ...], [$neuron0, ...], [$output0, ...], 
corresponding to the data from the last call to execute().
Intended primarily to assist with debugging.

=head2 get_internals

$network->get_internals()

Returns the weights in a not-human-consumable format.

=head2 readable

$network->readable()

Returns a human-friendly and diffable description of the network.

=head1 AUTHOR

Dan Collins <DCOLLINS@cpan.org>

=head1 COPYRIGHT AND LICENSE

This software is Copyright (c) 2011 by Dan Collins.

This is free software, licensed under:

  The GNU General Public License, Version 3, June 2007

=cut

